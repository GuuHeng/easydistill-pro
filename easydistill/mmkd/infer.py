
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
from tqdm import tqdm
from openai import OpenAI


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_field(filename):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        outputs = []
        for item in data:
            text = item["instruction"]
            image = item["image"]
            outputs.append((text, image))
        return outputs
    except FileNotFoundError:
        logging.error("The file was not found.")
    except json.JSONDecodeError:
        logging.error("There was an error decoding the JSON file.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def write_data_to_json_file(data, file_path):
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logging.info(f"Data successfully written to {file_path}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")


def generate_teacher_response_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)
    system_prompt = config["inference"]["system_prompt"]
    if system_prompt == "":
        system_prompt = "You are a helpful assistant."
    outcomes = []
    for text, image in tqdm(data_list, desc="Call remote model and generating responses"):
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image
                        },
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(
            messages = messages,
            model = model,
            max_completion_tokens = config["inference"]["max_new_tokens"]
        )
        result = completion.choices[0].message.content
        outcomes.append({'instruction': text, 'image': image, 'output': result})
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])


def infer_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_list = read_json_field(config["dataset"]["instruction_path"])
    try:
        job_type =  config["job_type"]
        if job_type == "mmkd_black_box_api":
            generate_teacher_response_api(data_list, config)
        else:
            logging.error(f"Invalid job type: {job_type}")
            raise ValueError(f"Invalid job type: {job_type}")
    except ValueError as e:
        logging.error(f"Training job terminated: {e}")
        return

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    infer_with_teacher_model(config)


if __name__ == "__main__":
    main()