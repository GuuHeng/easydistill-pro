
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
from jinja2 import Environment, BaseLoader, FileSystemLoader
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
import copy


def process_dataset(dataset_path, dataset_seed, env, template):
    examples = []
    with open(dataset_path, 'r') as file:
        examples = json.load(file)
    output_text = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    # use chat template
    for i in range(len(examples)):
        try:
            prompt_message = {"content": examples[i]["prompt"]}
            prompt = template.render(message=prompt_message, add_generation_prompt=False, add_output=False)

            chosen_message = {"content": examples[i]["prompt"], "output": examples[i]["chosen"]}
            chosen = template.render(message=chosen_message, add_generation_prompt=False, add_output=True)
            chosen = chosen[len(prompt):]

            rejected_message = {"content": examples[i]["prompt"], "output": examples[i]["rejected"]}
            rejected = template.render(message=rejected_message, add_generation_prompt=False, add_output=True)
            rejected = rejected[len(prompt):]

            output_text["prompt"].append(prompt)
            output_text["chosen"].append(chosen)
            output_text["rejected"].append(rejected)
        except:
            logging.warning(f"Error processing sample.")
            
    dataset = Dataset.from_dict(output_text)
    dataset = dataset.shuffle(seed=dataset_seed)        
    return dataset


def train(config):    
    dataset_path = config["dataset"]["labeled_path"]
    dataset_seed = config["dataset"]["seed"]
    
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    dataset = process_dataset(dataset_path, dataset_seed, env, template)
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True
    )    
    
    training_arguments = DPOConfig(**config["training"])    
    trainer = DPOTrainer(
        student_model,
        ref_model=copy.deepcopy(student_model),
        args=training_arguments,
        train_dataset=dataset,
        processing_class=student_tokenizer
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