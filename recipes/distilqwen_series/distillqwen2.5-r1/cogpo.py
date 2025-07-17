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

import trl
from trl.trainer.dpo_trainer import DataCollatorForPreference
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, FDivergenceConstants, FDivergenceType
from trl.trainer.utils import cap_exp
import json
import argparse


@dataclass
class DataCollatorForPreferenceWithBeta(DataCollatorForPreference):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        betas = torch.tensor([float(ex["beta"]) for ex in examples], dtype=torch.float32)
        for ex in examples:
            ex.pop("beta")         
        batch = super().torch_call(examples)
        batch["betas"] = betas
        return batch


class CogPOTrainer(DPOTrainer):
    def get_batch_loss_metrics(
        self,
        model,
        batch,
        train_eval: str = "train",  
    ):
        metrics = {}

        betas = batch.pop("betas").to(self.accelerator.device)  
        model_output = self.concatenated_forward(model, batch)

        if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
            ref_chosen_logps = batch["ref_chosen_logps"]
            ref_rejected_logps = batch["ref_rejected_logps"]
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

        losses, chosen_rewards, rejected_rewards = self._dpo_sigmoid_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            ref_chosen_logps,
            ref_rejected_logps,
            betas,                       
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        if self.args.rpo_alpha is not None:
            losses = losses + self.args.rpo_alpha * model_output["nll_loss"] 

        if self.use_weighting:
            losses = losses * model_output["policy_weights"]

        if self.aux_loss_enabled:
            losses = losses + self.aux_loss_coef * model_output["aux_loss"]

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
        metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
        metrics[f"{prefix}rewards/margins"] = (
            self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
        )
        metrics[f"{prefix}logps/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logps/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/chosen"] = (
            self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
        )
        metrics[f"{prefix}logits/rejected"] = (
            self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
        )
        if self.args.rpo_alpha is not None:
            metrics[f"{prefix}nll_loss"] = (
                self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
            )
        if self.aux_loss_enabled:
            metrics[f"{prefix}aux_loss"] = (
                self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
            )

        return losses.mean(), metrics

    def _dpo_sigmoid_loss(
        self,
        chosen_logps: torch.FloatTensor,
        rejected_logps: torch.FloatTensor,
        ref_chosen_logps: torch.FloatTensor,
        ref_rejected_logps: torch.FloatTensor,
        betas: torch.FloatTensor,              
    ):

        device = self.accelerator.device
        chosen_logratios = chosen_logps.to(device) - (not self.reference_free) * ref_chosen_logps.to(device)
        rejected_logratios = rejected_logps.to(device) - (not self.reference_free) * ref_rejected_logps.to(device)

        # 2) Δ = (log p_c - log p_r) - (log p̂_c - log p̂_r)
        if self.f_divergence_type == FDivergenceType.ALPHA_DIVERGENCE.value:
            alpha_coef = FDivergenceConstants.ALPHA_DIVERGENCE_COEF_DEFAULT
            if self.f_divergence_params and FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY in self.f_divergence_params:
                alpha_coef = float(self.f_divergence_params[FDivergenceConstants.ALPHA_DIVERGENCE_COEF_KEY])
            logits = (cap_exp(rejected_logratios * -alpha_coef) - cap_exp(chosen_logratios * -alpha_coef)) / alpha_coef
        else:
            logratios = chosen_logps - rejected_logps
            if self.reference_free:
                ref_logratios = torch.tensor([0], dtype=logratios.dtype, device=logratios.device)
            else:
                ref_logratios = ref_chosen_logps - ref_rejected_logps
            logratios = logratios.to(self.accelerator.device)
            ref_logratios = ref_logratios.to(self.accelerator.device)
            logits = logratios - ref_logratios

            if self.f_divergence_type == FDivergenceType.JS_DIVERGENCE.value:
                logits -= F.softplus(chosen_logratios) - F.softplus(rejected_logratios)


        losses = (
            -F.logsigmoid(betas * logits) * (1 - self.label_smoothing)
            - F.logsigmoid(-betas * logits) * self.label_smoothing
        )                                           

        chosen_rewards = betas * (chosen_logps.to(device) - ref_chosen_logps.to(device)).detach()
        rejected_rewards = betas * (rejected_logps.to(device) - ref_rejected_logps.to(device)).detach()
        

        return losses, chosen_rewards, rejected_rewards


def train(config):
    model_name = config["models"]["student"]
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"], split='train')

    dpo_args = DPOConfig(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["num_train_epochs"],
        loss_type=config["training"]["loss_type"],
        beta=config["training"]["beta"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        remove_unused_columns=False,
    )

    collator = DataCollatorForPreferenceWithBeta(
        pad_token_id=tokenizer.pad_token_id
    )

    trainer = CogPOTrainer(
        model=model,
        args=dpo_args,
        train_dataset=dataset,
        tokenizer=tokenizer,         
        data_collator=collator,
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
