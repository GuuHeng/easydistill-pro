
# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import json
import argparse
import logging
import os
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardTrainer, RewardConfig
from datasets import Dataset


def process_dataset(dataset_path, tokenizer, config, template):
    kwargs = {"padding": "max_length", "truncation": True, "max_length": config["training"]["max_length"], "return_tensors": "pt"}
    examples = []
    try:
        with open(dataset_path, 'r') as file:
            examples = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file '{dataset_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{dataset_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(examples)
    output_dataset = []
    # use chat template
    for i in range(len(examples)):
        try:
            chosen_message = {"content": examples[i]["prompt"], "output": examples[i]["chosen"]}
            prompt_plus_chosen_response = template.render(message=chosen_message, add_generation_prompt=False, add_output=True)
                        
            rejected_message = {"content": examples[i]["prompt"], "output": examples[i]["rejected"]}
            prompt_plus_rejected_response = template.render(message=rejected_message, add_generation_prompt=False, add_output=True)
            
            tokens_chosen = tokenizer.encode_plus(prompt_plus_chosen_response, **kwargs)
            tokens_rejected = tokenizer.encode_plus(prompt_plus_rejected_response, **kwargs)
            sample = {
                "input_ids_chosen": tokens_chosen["input_ids"][0], "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                "input_ids_rejected": tokens_rejected["input_ids"][0], "attention_mask_rejected": tokens_rejected["attention_mask"][0]
            }
            output_dataset.append(sample)
        except:
            logging.warning(f"Error processing sample.")
    dataset = Dataset.from_list(output_dataset)
    return dataset


def train(config):
    dataset_path = config["dataset"]["labeled_path"]
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    dataset = process_dataset(dataset_path, student_tokenizer, config, template)

    student_model = AutoModelForSequenceClassification.from_pretrained(
        config["models"]["student"],
        num_labels=1,
        trust_remote_code=True
    )
    student_model.config.pad_token_id = student_tokenizer.pad_token_id

    training_arguments = RewardConfig(**config["training"])    
    trainer = RewardTrainer(
        model=student_model,
        processing_class=student_tokenizer,
        args=training_arguments,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    student_tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)


if __name__ == "__main__":
    main()