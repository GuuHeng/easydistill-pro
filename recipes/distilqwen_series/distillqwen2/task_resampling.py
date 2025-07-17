
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
import re
import logging
from openai import OpenAI
from collections import Counter
import random
import argparse


predefined_distribution = {
    'Math': 0.167,
    'Code Generation': 0.083,
    'Writing': 0.017,
    'Computer Science': 0.017,
    'Reasoning': 0.167,
    'Complex Format': 0.017,
    'Code Debug': 0.083,
    'Common-Sense': 0.017,
    'Counterfactual': 0.017,
    'Multilingual': 0.017,
    'Roleplay': 0.017,
    'Biology': 0.017,
    'Technology': 0.017,
    'Ethics': 0.017,
    'Sport': 0.017,
    'Law': 0.017,
    'Medicine': 0.017,
    'Literature': 0.017,
    'Entertainment': 0.017,
    'Art': 0.017,
    'Music': 0.017,
    'Toxicity': 0.017,
    'Economy': 0.017,
    'Physics': 0.017,
    'History': 0.017,
    'Chemistry': 0.017,
    'Philosophy': 0.017,
    'Health': 0.017,
    'Ecology': 0.017,
    'Grammar': 0.017,
    'Paraphrase': 0.017,
    'Others': 0.041
}

predefined_prompt = """
You are a data annotation expert. Please classify the task type or domain of #Given Instruction. 
The task type or domain should be in the list: [’Math’, ’Code Generation’, ’Writing’, ’Computer Science’, ’Reasoning’, ’Complex Format’, ’Code Debug’, ’Common-Sense’, ’Counterfactual’, ’Multilingual’, ’Roleplay’,’Biology’, ’Technology’, ’Ethics’, ’Sport’, ’Law’, ’Medicine’, ’Literature’, ’Entertainment’, ’Art’, ’Music’, ’Toxicity’, ’Economy’, ’Physics’, ’History’, ’Chemistry’, ’Philosophy’,’Health’,’Ecology’,’Grammar’,’Paraphrase’, ’Others’]. You should place your answer enclosed within <answer></answer> tags, such as <answer>Math</answer>. Do not return anything else.
#Given Instruction#:
"""


def extract_answer(content):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        return None
    

def classify_instruction(instruction, client, model):
    message = [
        {"role": "user", "content": predefined_prompt + "\n" + instruction}
    ]
    completion = client.chat.completions.create(
        messages = message,
        model = model,
        max_completion_tokens = 1024
    )
    result = completion.choices[0].message.content.strip()
    print(result)
    result = extract_answer(result)
    if result is None or result not in predefined_distribution.keys():
        result = 'Others'
    print(result)
    return result
    
    
def main(args):
    # Load dataset
    with open(args.input_file, 'r') as file:
        data = json.load(file)

    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )
    models = client.models.list()
    model = models.data[0].id
    logging.info(model)

    # Classify each instruction
    classified_data = []
    count = 0
    for item in data:
        category = classify_instruction(item['instruction'], client, model)
        classified_data.append({'instruction': item['instruction'], 'category': category})
        count += 1
        print(count)

    # Count occurrences per category
    category_counts = Counter(item['category'] for item in classified_data)
    total_samples = len(classified_data)

    # Resample according to predefined distribution
    resampled_data = []
    for category, target_ratio in predefined_distribution.items():
        target_count = int(total_samples * target_ratio)
        category_samples = [item for item in classified_data if item['category'] == category]
        if len(category_samples) == 0:
            logging.warning("No instructions are provided for the category: " + category)
            continue
        if len(category_samples) > target_count:
            print(category)
            print(len(category_samples))
            print(target_count)
            # Randomly sample the required number of instructions
            resampled_category_samples = random.sample(category_samples, target_count)
        else:
            # If not enough samples, repeat the existing ones
            resampled_category_samples = category_samples * (target_count // len(category_samples)) + random.sample(category_samples, target_count % len(category_samples))
        resampled_data.extend(resampled_category_samples)

    # Save final dataset
    with open(args.output_file, 'w') as file:
        json.dump(resampled_data, file, indent=4)

    print("Resampling complete. Final output saved to '{}'.".format(args.output_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Task and Domain Classification')
    parser.add_argument('--input-file', type=str, required=True, help='Input JSON file containing instructions.')
    parser.add_argument('--output-file', type=str, required=True, help='Output JSON file to store resampled instructions.')
    parser.add_argument('--api-key', type=str, required=True, help='API key.')
    parser.add_argument('--base-url', type=str, required=True, help='Base URL.')

    args = parser.parse_args()
    main(args)
