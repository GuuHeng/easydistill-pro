
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

import jsonlines
import logging
import os
from jinja2 import Environment, FileSystemLoader
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI

from utils import write_data_to_json_file


# I have checked this function.
def cot_generate_api(data_list, config):
    client = OpenAI(
        api_key = config["inference"]["api_key"],
        base_url = config["inference"]["base_url"]
    )
    models = client.models.list()
    model = models.data[0].id
    prompt = config["inference"]["prompt"]
    stream = config["inference"]["stream"]
    logging.info(model)
    outcomes = []
    for sample in tqdm(data_list, desc="Calling remote model and generating responses"):
        sample = prompt + "\n" + sample
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
        if result is not None:
            outcomes.append({"instruction": sample, "output": result})
    write_data_to_json_file(outcomes, config["dataset"]["output_path"])


def cot_generate_batch(tokenizer, llm, data_list, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    prompt = config["inference"]["prompt"]

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
        outcomes = []
        for i in range(len(batch)):
            if responses[i] is not None:
                outcomes.append((sample,responses[i]))

        with jsonlines.open(config["dataset"]["output_path"], mode='a') as writer:
            for ins,result in outcomes:
                gen_data = {"instruction": ins, "output": result}
                writer.write(gen_data)


def cot_long2short_api(data_list_ins, data_list_out, config):
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
    data_list=[(ins,out) for ins,out in zip(data_list_ins,data_list_out)]
    for ins,out in tqdm(data_list, desc="Calling remote model and generating responses"):
        sample = f"{prompt} Simplify the reasoning process for the problem below.\n\nProblem:\n{ins}\n\nAnswer:\n{out}\n\nSimplified Reasoning Process:"
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
        
        if result is not None:
            outcomes.append((sample,result))

    with jsonlines.open(config["dataset"]["output_path"], mode='a') as writer:
        for ins,result in outcomes:
            gen_data = {"instruction": ins, "output": result}
            writer.write(gen_data)


def cot_long2short_batch(tokenizer, llm, data_list_ins, data_list_out, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    prompt = config["inference"]["prompt"]
    data_list=[(ins,out) for ins,out in zip(data_list_ins,data_list_out)]
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for ins,out in batch:
            sample = f"{prompt} Simplify the reasoning process for the problem below.\n\nProblem:\n{ins}\n\nAnswer:\n{out}\n\nSimplified Reasoning Process:"
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
        outcomes = []
        for i in range(len(batch)):
            if responses[i] is not None:
                outcomes.append((sample,responses[i]))

        with jsonlines.open(config["dataset"]["output_path"], mode='a') as writer:
            for ins,result in outcomes:
                gen_data = {"instruction": ins, "output": result}
                writer.write(gen_data)


def cot_short2long_api(data_list_ins, data_list_out, config):
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
    data_list=[(ins,out) for ins,out in zip(data_list_ins,data_list_out)]
    for ins,out in tqdm(data_list, desc="Calling remote model and generating responses"):
        sample = f"{prompt} Extend the reasoning process for the problem below.\n\nProblem:\n{ins}\n\nAnswer:\n{out}\n\nExtended Reasoning Process:"
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
        
        if result is not None:
            outcomes.append((sample,result))

    with jsonlines.open(config["dataset"]["output_path"], mode='a') as writer:
        for ins,result in outcomes:
            gen_data = {"instruction": ins, "output": result}
            writer.write(gen_data)


def cot_short2long_batch(tokenizer, llm, data_list_ins, data_list_out, config, batch_size=32):
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    prompt = config["inference"]["prompt"]
    data_list=[(ins,out) for ins,out in zip(data_list_ins,data_list_out)]
    batches = [data_list[i:i + batch_size] for i in range(0, len(data_list), batch_size)]
    for batch in tqdm(batches, desc="Generating responses"):
        new_batch = []
        for ins,out in batch:
            sample = f"{prompt} Extend the reasoning process for the problem below.\n\nProblem:\n{ins}\n\nAnswer:\n{out}\n\nExtended Reasoning Process:"
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
        outcomes = []
        for i in range(len(batch)):
            if responses[i] is not None:
                outcomes.append((sample,responses[i]))

        with jsonlines.open(config["dataset"]["output_path"], mode='a') as writer:
            for ins,result in outcomes:
                gen_data = {"instruction": ins, "output": result}
                writer.write(gen_data)