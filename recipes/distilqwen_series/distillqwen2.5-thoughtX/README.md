# DistilQwen-ThoughtX: Optimized Reasoning Models with OmniThought

## Brief Introduction

DistilQwen-ThoughtX is a series of high-performance reasoning models trained on the [OmniThought](https://huggingface.co/datasets/alibaba-pai/OmniThought) dataset. These models are optimized for chain-of-thought (CoT) reasoning with balanced verbosity and cognitive difficulty, achieving state-of-the-art results on mathematical, coding, and logical reasoning benchmarks.

## Detailed Steps

### Direct Training

DistilQwen-ThoughtX was trained using data from the OmniThought dataset, which includes 2 million CoT processes with RV (Reasoning Verbosity) and CD (Cognitive Difficulty) annotations. The dataset covers mathematics, coding, and logical reasoning tasks, validated by multiple teacher models (DeepSeek-R1, QwQ-32B).

The training system prompt is:

```json
{
    "system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
}
```

Using the OmniThought dataset, we can run the training job:

```bash
python easydistill/kd/train.py --config=distilqwen2.5-thoughtx-train.json
```

Remember to filter the RV and CD annotations to ensure they are within the desired range to train your own model.

| Model Name                           | Parameters | Base Model          |
|--------------------------------------|------------|---------------------|
| `DistilQwen-ThoughtX-7B`             | 7B         | Qwen2.5-7B-Instruct | 
| `DistilQwen-ThoughtX-32B`            | 32B        | Qwen2.5-32B-Instruct|

### Process Your Own Data

To obtain the RV and CD values of your own data, you can use the following prompt to call QwQ-32B/Deepseek-R1, score your own data, and filter it.

Prompt Template to Calculate the RV Score：
```json
{
    "prompt": "You are an expert judge tasked with evaluating the Reasoning Verbosity of a Chain-of-Thought (CoT) for a given problem and its answer. Reasoning Verbosity Evaluation Focus: Assess how well the CoT’s length and step complexity match the problem’s inherent difficulty. An optimal chain is neither missing essential steps nor padded with needless digressions. A simple question should be solved with a brief, direct chain; a challenging one may justifiably require a longer path with reflection and error-checking. Scoring Guidelines (0-9): 0-1 Minimal verbosity, straightforward expression with little to no elaboration. 2-3 Clear and concise reasoning with necessary explanations. 4-5 Moderate verbosity with detailed explanations and thorough reasoning. 6-7 Extensive verbosity with comprehensive justification and exploration of complex connections. 8-9 High verbosity with deep, exhaustive exploration of reasoning; involves extensive elaboration, nested justifications, and consideration of counterarguments or alternative perspectives. Given Problem, Chain-of-Thought and Answer, you will: 1. Analyze the Reasoning Verbosity 2. Determine score using the above criteria 3. Output ONLY the integer score (0-9) Problem: {problem} Chain-of-Thought: {thought} Answer: {solution}"
}
```

Prompt Template to Calculate the CD Score：
```json
{
    "prompt": "You are an expert judge assessing the Cognitive Difficulty of a Chain-of-Thought (CoT) for a given problem and its answer. Cognitive Difficulty Evaluation Focus: The level of reasoning competence required for a model to follow and reproduce the chain faithfully. Judge the reasoning approach, techniques, and overall difficulty. Higher scores correspond to more advanced concepts, abstractions, or multi-layer reasoning patterns. Scoring Guidelines (0-9): 0-1 Elementary facts or a single trivial operation. 2-3 Multi-step arithmetic, explicit enumeration, basic rule chaining. 4-5 Early-undergraduate logic/algebra; one non-obvious insight. 6-7 Advanced undergraduate techniques (determinants, dynamic programming, layered code reasoning, etc). 8-9 Graduate-level abstraction, nested proofs, intricate algorithmic analysis. Given Problem, Chain-of-Thought and Answer, you will: 1. Analyze the Cognitive Difficulty 2. Determine score using the above criteria 3. Output ONLY the integer score (0-9) Problem: {problem} Chain-of-Thought: {thought} Answer: {solution}"
}
```

## Model Download

We have open-sourced our distilled models on HuggingFace. The available models are named `alibaba-pai/DistilQwen-ThoughtX-7B` and `alibaba-pai/DistilQwen-ThoughtX-32B`.

Users can download these models from HuggingFace using the following code:

```python
from huggingface_hub import snapshot_download

# Download the 7B model
model_name = "alibaba-pai/DistilQwen-ThoughtX-7B"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen-ThoughtX-7B/")

# Download the 32B model
model_name = "alibaba-pai/DistilQwen-ThoughtX-32B"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen-ThoughtX-32B/")
```

## Performance

The models achieve state-of-the-art performance on various reasoning benchmarks:

| Model                | AIME2024 | MATH500 | GPQA-D | LiveCodeBench V2 |
|----------------------|----------|---------|--------|------------------|
| DeepSeek-R1-Distill-7B | 57.3     | 89.6    | 47.3   | 48.4             |
| **DistilQwen-ThoughtX-7B** | **56.7** | **90.2** | **50.0** | **56.8**         |
| DeepSeek-R1-Distill-32B | 74.7     | 90.0    | 62.4   | 72.3             |
| **DistilQwen-ThoughtX-32B** | **80.0** | **92.6** | **64.0** | **73.4**         |

## Reference

For more detailed information about the model, we encourage you to refer to our paper:

- **Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations**  
  Wenrui Cai, Chengyu Wang, Junbing Yan, Jun Huang, Xiangzhong Fang
  [arXiv:2505.10937](https://arxiv.org/abs/2505.10937)

You can cite the paper using the following citation format:

```bibtex
@misc{cai2025reasoningomnithoughtlargecot,
      title={Reasoning with OmniThought: A Large CoT Dataset with Verbosity and Cognitive Difficulty Annotations}, 
      author={Wenrui Cai and Chengyu Wang and Junbing Yan and Jun Huang and Xiangzhong Fang},
      year={2025},
      eprint={2505.10937},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.10937} 
}
``` 