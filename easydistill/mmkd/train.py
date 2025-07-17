
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
from datasets import load_dataset, Dataset
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
from trl import SFTTrainer, SFTConfig


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train(config):
    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"])
    dataset = dataset.shuffle(seed=config["dataset"]["seed"])["train"]
    student_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(config["models"]["student"])

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            chat = [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image","image": example["image"]
                        }, 
                        {
                            "type": "text","text": example["instruction"]
                        }
                    ]
                },
                {
                    "role": "assistant", 
                    "content": example["output"]
                }
            ]
            text = processor.apply_chat_template(chat, tokenize=False)
            texts.append(text)
            image, _ = process_vision_info(chat)
            images.append(image)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        
        if isinstance(processor, Qwen2_5_VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch
    
    training_arguments = SFTConfig(**config["training"])
    training_arguments.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_arguments.remove_unused_columns = False
    training_arguments.dataset_kwargs = {"skip_prepare_dataset": True}
    
    trainer = SFTTrainer(
        model=student_model,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
        args=training_arguments,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    processor.tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)


if __name__ == "__main__":
    main()