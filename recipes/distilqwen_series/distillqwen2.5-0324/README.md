# DistilQwen2.5-0324: training fast-thinking models

## Brief Introduction

In the rapid advancement of large language models, effectively balancing the trade-off between efficient inference and model thinking capabilities has been a key focus in both academia and industry. DeepSeekV3-0324, by default, does not employ deep thinking mode, which accelerates model inference while maintaining a balance between swift reasoning and handling complex tasks. The DistilQwen2.5-0324 series not only inherits the essence of the original model's chain-of-thought distillation but also introduces fast-thinking strategies, significantly boosting inference speed. This enables these models to efficiently execute complex tasks on resource-constrained devices and in edge computing scenarios.

## Detailed Steps

### Processing of Instructional Dataset

DistilQwen2.5-0324 was trained using data distilled from Deepseek-V3-0324 as well as data rewritten with long2short after distillation from Deepseek-R1. For Deepseek-V3-0324, the official recommendation is not to use a system prompt; for the long2short scenario, the following prompt was used. You can employ this method to reduce the output of Deepseek-R1 and distill your own model.

```json
{
    "system": "You are a helpful assistant who is highly skilled at simplifying reasoning processes. Given a problem, its answer and its reasoning process, your task is to simplify the reasoning process so that a small language model (e.g., a 7B model) can reliably follow the steps to solve the problem. If the original reasoning process is divided into multiple steps separated by two newline characters (\n\n), your output must preserve this formatting. You must output ONLY the simplified reasoning process with no additional explanation or commentary."
}
```

```bash
python easydistill/kd/infer.py --config=distilqwen2.5-0324_stage1.json
```

The training dataset is in JSON format, exemplified by entries such as:

```json
[
    {
        "instruction": "The ratio of the number of molar teeth in the human upper jaw at the age of 6 is 2:1 compared to number of incisors teeth. There are total 8 incisors in the human mouth...",
        "output": "Step 1: Determine the total number of incisors in the upper jaw...The final answer is: \\boxed{8}"
    }
]
```

### Black-Box KD

The black-box KD process follows a supervised learning paradigm, utilizing enhanced instruction-response pairs as training samples. Through this approach, the student model can effectively absorb and understand the knowledge imparted by the larger model, even with a limited number of parameters. This method not only boosts the student model's ability to tackle tasks but also enables it to perform better in multi-task scenarios. Because we have already obtained the teacher's responses in the dataset, we can run the training job:

```bash
python easydistill/kd/train.py --config=distilqwen2.5-0324_stage2.json
```

Plese refer to the config file `distilqwen2.5-0324_stage2.json` in the current folder. If you need to run the job in a distributed mode, use `accelerate` to run the job.

## Model Download

We have open-sourced our distilled models on both HuggingFace and ModelScope. The available models are named `alibaba-pai/DistilQwen2.5-DS3-0324-7B`, `alibaba-pai/DistilQwen2.5-DS3-0324-14B`, and `alibaba-pai/DistilQwen2.5-DS3-0324-32B`.

For example, users can download these models from HuggingFace using the following code:


```python
from huggingface_hub import snapshot_download

# Download the 1.5B model
model_name = "alibaba-pai/DistilQwen2.5-DS3-0324-7B"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-DS3-0324-7B/")

# Download the 3B model
model_name = "alibaba-pai/DistilQwen2.5-DS3-0324-14B"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-DS3-0324-14B/")

# Download the 7B model
model_name = "alibaba-pai/DistilQwen2.5-DS3-0324-32B"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-DS3-0324-32B/")
```


## Performance

- **32B Model** approaches the performance of closed-source models with 10x the parameters on the GPQA Diamond benchmark
- **Significant Improvement in Reasoning Efficiency** (see comparison table below)

| Model                          | MMLU_PRO Tokens | AIME2024 Tokens | Speed Gain |
|--------------------------------|-----------------|-----------------|------------|
| DistilQwen2.5-R1-32B (Slow-Thinking) | 4198            | 12178           | 1x         |
| DistilQwen2.5-DS3-0324-32B     | 690             | 4177            | 5-8x       |