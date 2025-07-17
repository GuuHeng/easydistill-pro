
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

import argparse
import logging
import json

from instruct_synthesis import (
    expand_instruction_api,
    expand_instruction_batch,
    refine_instruction_api,
    refine_instruction_batch,
    instruction_response_extraction_api,
    instruction_response_extraction_batch
)
from cot_synthesis import (
    cot_generate_api,
    cot_generate_batch,
    cot_long2short_api,
    cot_long2short_batch,
    cot_short2long_api,
    cot_short2long_batch
)
from utils import read_json_field, load_tokenizer_and_vllm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def data_synthesis_with_teacher_model(config):
    logging.info('Generating distillation data from the teacher model!')
    job_type =  config["job_type"]
    if job_type == "instruction_response_extraction_api":
        data_list = read_json_field(config["dataset"]["input_path"], field_name="data")
    elif job_type in ["cot_long2short_api","cot_long2short_batch","cot_short2long_api","cot_short2long_batch"]:
        data_list_ins = read_json_field(config["dataset"]["input_path"])
        data_list_out = read_json_field(config["dataset"]["input_path"], field_name="output")
    else:
        data_list = read_json_field(config["dataset"]["input_path"])

    try:
        if job_type == "instruction_expansion_api":
            expand_instruction_api(data_list, config)
        elif job_type == "instruction_expansion_batch":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            expand_instruction_batch(tokenizer, llm, data_list, config)

        elif job_type == "instruction_refinement_api":
            refine_instruction_api(data_list, config)
        elif job_type == "instruction_refinement_batch":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            refine_instruction_batch(tokenizer, llm, data_list, config)

        elif job_type == "instruction_response_extraction_api":
            instruction_response_extraction_api(data_list, config)
        elif job_type == "instruction_response_extraction_batch":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            instruction_response_extraction_batch(tokenizer, llm, data_list, config)

        elif job_type == "cot_generation_api":
            cot_generate_api(data_list, config)
        elif job_type == "cot_generation_batch":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            cot_generate_batch(tokenizer, llm, data_list, config)

        elif job_type == "cot_long2short_api":
            cot_long2short_api(data_list_ins, data_list_out, config)
        elif job_type == "cot_long2short_batch":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            cot_long2short_batch(tokenizer, llm, data_list_ins, data_list_out, config)

        elif job_type == "cot_short2long_api":
            cot_short2long_api(data_list_ins, data_list_out, config)
        elif job_type == "cot_short2long_batch":
            tokenizer, llm = load_tokenizer_and_vllm(config)
            cot_short2long_batch(tokenizer, llm, data_list_ins, data_list_out, config)
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
    data_synthesis_with_teacher_model(config)


if __name__ == "__main__":
    main()