
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
import torch
import logging
from vllm import LLM
from transformers import AutoTokenizer


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