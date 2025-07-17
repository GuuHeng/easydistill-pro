
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
import random
from jinja2 import Environment, BaseLoader, FileSystemLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from trl import PPOConfig, PPOTrainer


def process_dataset(dataset_path, dataset_seed, env, template, tokenizer, train_ratio):
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

    output_dataset = []
    # use chat template
    for i in range(len(examples)):
        try:
            message = {"content": examples[i]["instruction"]}
            rendered = template.render(message=message, add_generation_prompt=True, add_output=False)
            tokens = tokenizer.encode(rendered)
            sample = {"input_ids": tokens}
            output_dataset.append(sample)
        except:
            logging.warning(f"Error processing sample.")
            
    random.shuffle(output_dataset)
    random.seed(dataset_seed)
    split_index = int(len(output_dataset) * train_ratio)
    train_list = output_dataset[:split_index]
    eval_list = output_dataset[split_index:]
    
    return Dataset.from_list(train_list), Dataset.from_list(eval_list)


def train(config):    
    dataset_path = config["dataset"]["instruction_path"]
    dataset_seed = config["dataset"]["seed"]
    train_ratio = config["dataset"]["train_ratio"]
    
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    train_dataset, eval_dataset = process_dataset(dataset_path, dataset_seed, env, template, tokenizer, train_ratio)
    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    print(train_dataset)
    print(eval_dataset)
    
    reward_model_path = config["models"]["reward"]
    sft_model_path = config["models"]["student"]
    value_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, trust_remote_code=True, num_labels=1
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path, trust_remote_code=True, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        sft_model_path, trust_remote_code=True
    )
    policy = AutoModelForCausalLM.from_pretrained(
        sft_model_path, trust_remote_code=True
    )

    training_arguments = PPOConfig(**config["training"])
    trainer = PPOTrainer(
        config=training_arguments,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)


if __name__ == "__main__":
    main()