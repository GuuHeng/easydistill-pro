
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

import logging
import os
from jinja2 import Environment, FileSystemLoader
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI
import random
import re

from utils import read_json_field, write_data_to_json_file, load_tokenizer_and_vllm


def extract_answer(content):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    
    
def extract_instruction_response(content):
    instruction_pattern = r'<instruction>(.*?)</instruction>'
    instruction_match = re.search(instruction_pattern, content, re.DOTALL)
    response_pattern = r'<response>(.*?)</response>'
    response_match = re.search(response_pattern, content, re.DOTALL)
    if instruction_match and response_match:
        return instruction_match.group(1), response_match.group(1)
    else:
        return None, None
    

def generate_prompt_list(data_list, prompt, num_in_context_samples, num_output_samples):
    if num_in_context_samples > len(data_list):
        raise ValueError("num_in_context_samples cannot be larger than the length of data_list")
    output_list = []
    for _ in range(num_output_samples):
        selected_samples = random.sample(data_list, num_in_context_samples)
        combined_prompts = prompt + "\n" + "".join([sample + "\n" for sample in selected_samples])
        output_list.append(combined_prompts)
    return output_list
            
            
def expand_instruction_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"],
    )
    models = client.models.list()
    model = models.data[0].id
    num_output_samples = config["dataset"]["num_output_samples"]
    num_in_context_samples = config["dataset"]["num_in_context_samples"]
    prompt = config["inference"]["prompt"]
    stream = config["inference"]["stream"]
    logging.info(model)
    prompt_list = generate_prompt_list(data_list, prompt, num_in_context_samples, num_output_samples)
    outcomes = []
    for sample in tqdm(prompt_list, desc="Calling remote model and generating responses"):
        logging.info(sample)
        message = [
            {"role": "user", "content": sample}
        ]
        completion = client.chat.completions.create(
            messages = message,
            model = model,
            max_completion_tokens = config["inference"]["max_new_tokens"],
            stream = stream,
        )
        if stream:
            result = ""
            for chunk in completion:
                result += chunk.choices[0].delta.content
        else:
            result = completion.choices[0].message.content
        result = extract_answer(result)
        if result is not None:
            outcomes.append({"instruction": result})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])


def expand_instruction_batch(tokenizer, llm, data_list, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)

    num_output_samples = config["dataset"]["num_output_samples"]
    num_in_context_samples = config["dataset"]["num_in_context_samples"]
    prompt = config["inference"]["prompt"]
    prompt_list = generate_prompt_list(data_list, prompt, num_in_context_samples, num_output_samples)

    outcomes = []
    batches = [prompt_list[i:i + batch_size] for i in range(0, len(prompt_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            logging.info(sample)
            message={"role": "user", "content": sample}
            full_text = template.render(
                message=message,
                add_generation_prompt=True,
                add_output=False
            )
            new_batch.append(full_text)
        outputs = llm.generate(
            new_batch,
            SamplingParams(
                n=1,
                top_k=1,
                temperature=config["inference"]["temperature"],
                seed=config["inference"]["seed"],
                skip_special_tokens=False,
                ignore_eos=False,
                max_tokens=config["inference"]["max_new_tokens"]
            )
        )
        responses = [output.outputs[0].text for output in outputs]
        for i in range(len(batch)):
            result = extract_answer(responses[i])
            if result is not None:
                outcomes.append({"instruction": result})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])
                

def refine_instruction_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"],
    )
    models = client.models.list()
    model = models.data[0].id
    prompt = config["inference"]["prompt"]
    stream = config["inference"]["stream"]
    logging.info(model)
    outcomes = []
    for sample in tqdm(data_list, desc="Calling remote model and generating responses"):
        sample = prompt + "\n" + sample
        logging.info(sample)
        message = [
            {"role": "user", "content": sample}
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
        result = extract_answer(result)
        if result is not None:
            outcomes.append({"instruction": result})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])


def refine_instruction_batch(tokenizer, llm, data_list, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    prompt = config["inference"]["prompt"]

    outcomes = []
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            sample = prompt + "\n" + sample
            logging.info(sample)
            message={"role": "user", "content": sample}
            full_text = template.render(
                message=message,
                add_generation_prompt=True,
                add_output=False
            )
            new_batch.append(full_text)
        outputs = llm.generate(
            new_batch,
            SamplingParams(
                n=1,
                top_k=1,
                temperature=config["inference"]["temperature"],
                seed=config["inference"]["seed"],
                skip_special_tokens=False,
                ignore_eos=False,
                max_tokens=config["inference"]["max_new_tokens"],
            )
        )
        responses = [output.outputs[0].text for output in outputs]
        for i in range(len(batch)):
            result = extract_answer(responses[i])
            if result is not None:
                outcomes.append({"instruction": result})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])


def instruction_response_extraction_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"],
    )
    models = client.models.list()
    model = models.data[0].id
    prompt = config["inference"]["prompt"]
    stream = config["inference"]["stream"]
    logging.info(model)
    outcomes = []
    for sample in tqdm(data_list, desc="Calling remote model and generating responses"):
        sample = prompt + "\n" + sample
        logging.info(sample)
        message = [
            {"role": "user", "content": sample}
        ]
        completion = client.chat.completions.create(
            messages = message,
            model = model,
            max_completion_tokens = config["inference"]["max_new_tokens"],
            stream=  stream,
        )
        if stream:
            result = ""
            for chunk in completion:
                result += chunk.choices[0].delta.content
        else:
            result = completion.choices[0].message.content
        new_instruction, new_response = extract_instruction_response(result)
        if new_instruction is not None and new_response is not None:
            outcomes.append({"instruction": new_instruction, "output": new_response})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])

            
def instruction_response_extraction_batch(tokenizer, llm, data_list, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    prompt = config["inference"]["prompt"]

    outcomes = []
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for sample in batch:
            logging.info(sample)
            sample = prompt + "\n" + sample
            message={"role": "user", "content": sample}
            full_text = template.render(
                message=message,
                add_generation_prompt=True,
                add_output=False
            )
            new_batch.append(full_text)
        outputs = llm.generate(
            new_batch,
            SamplingParams(
                n=1,
                top_k=1,
                temperature=config["inference"]["temperature"],
                seed=config["inference"]["seed"],
                skip_special_tokens=False,
                ignore_eos=False,
                max_tokens=config["inference"]["max_new_tokens"],
            )
        )
        responses = [output.outputs[0].text for output in outputs]
        for i in range(len(batch)):
            new_instruction, new_response = extract_instruction_response(responses[i])
            if new_instruction is not None and new_response is not None:
                outcomes.append({"instruction": new_instruction, "output": new_response})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])