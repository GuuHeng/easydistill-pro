
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

import json, jsonlines
import argparse
import torch
import logging
import os
from jinja2 import Environment, FileSystemLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI
import math


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def read_json_field(filename, field_name='instruction'):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        output_fields = []
        for item in data:
            if field_name in item:
                output_fields.append(item[field_name])
        return output_fields
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


def load_tokenizer_and_vllm(config, eos_token=None):
    teacher_model_path = config["models"]["teacher"]
    logging.info(f"Loading ckpt and tokenizer: {teacher_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if eos_token:
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        logging.info(f"eos_token {eos_token} from user input")
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
        logging.info(f"Initial eos_token_id {tokenizer.eos_token_id} from tokenizer")
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
    else:
        raise ValueError("No available eos_token or eos_token_id.")
    try:
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_token_id
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
    except:
        logging.info(f"[WARNING] Cannot set tokenizer.eos_token")
    logging.info(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    logging.info(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")
    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=teacher_model_path,
        tensor_parallel_size=num_gpus,
        enable_chunked_prefill=config["inference"]["enable_chunked_prefill"],
        gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
        trust_remote_code=config["inference"]["trust_remote_code"],
        dtype=torch.bfloat16,
        enforce_eager=config["inference"]["enforce_eager"],
        max_model_len=config["inference"]["max_model_len"],
    )
    logging.info("vLLM model loaded successfully")
    return tokenizer, llm


def generate_teacher_response_batch(tokenizer, llm, data_list, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    outcomes = []
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            message = {"role": "user", "content": sample}
            full_text = template.render(
                message = message,
                add_generation_prompt = True,
                add_output = False
            )
            new_batch.append(full_text)
        outputs = llm.generate(
            new_batch,
            SamplingParams(
                n = 1,
                top_k = 1,
                temperature = config["inference"]["temperature"],
                seed = config["inference"]["seed"],
                skip_special_tokens = False,
                ignore_eos = False,
                max_tokens = config["inference"]["max_new_tokens"]
            )
        )
        responses = [output.outputs[0].text for output in outputs]
        gen_data = [{'instruction': batch[i], 'output': responses[i]} for i in range(len(batch))]
        outcomes = outcomes + gen_data
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])


def generate_teacher_logits_batch(tokenizer, llm, data_list, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)

    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            message={"role": "user", "content": sample}
            full_text = template.render(
                message=message,
                add_generation_prompt=True,
                add_output=False
            )
            new_batch.append(full_text)
        
        outputs = llm.generate(
            new_batch,  # Pass the raw text directly
            SamplingParams(
                n=1,
                top_k=1,
                temperature=config["inference"]["temperature"],
                seed=config["inference"]["seed"],
                skip_special_tokens=False,
                ignore_eos=True,
                max_tokens=config["inference"]["max_new_tokens"],
                logprobs=config["inference"]["top_logits_num"],
            )
        )
        # Extract the generated logits
        responses = [output.outputs[0].text for output in outputs]
        logits=[output.outputs[0].logprobs for output in outputs]
        for logit in logits:
            for pos in logit:
                for k,v in pos.items():
                    pos[k]=math.exp(v.logprob)
        
        with jsonlines.open(config["dataset"]["logits_path"], mode='a') as writer:
            for row in logits:
                #for item in row:
                writer.write(row)


def generate_teacher_response_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)
    system_prompt = config["inference"]["system_prompt"]
    stream = config["inference"]["stream"]
    outcomes = []
    for sample in tqdm(data_list, desc="Call remote model and generating responses"):
        if system_prompt == "":
            message = [
                {'role': 'user', 'content': sample}
            ]
        else:
            message = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': sample}
            ]
        completion = client.chat.completions.create(
            messages = message,
            model = model,
            max_completion_tokens = config["inference"]["max_new_tokens"],
            stream = stream
        )
        if stream:
            result = ""
            for chunk in completion:
                result += chunk.choices[0].delta.content
        else:
            result = completion.choices[0].message.content
            
        outcomes.append({'instruction': sample, 'output': result})
    write_data_to_json_file(outcomes, config["dataset"]["labeled_path"])


def infer_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    data_list = read_json_field(config["dataset"]["instruction_path"])
    try:
        job_type =  config["job_type"]
        if job_type == "kd_black_box_api":
            generate_teacher_response_api(data_list, config)
        elif job_type == "kd_black_box_local":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            generate_teacher_response_batch(tokenizer, llm, data_list, config)
        elif job_type == "kd_white_box":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            generate_teacher_logits_batch(tokenizer, llm, data_list, config)
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