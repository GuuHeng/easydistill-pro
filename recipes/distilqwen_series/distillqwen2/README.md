# DistilQwen2: Refining Instructional Data for Black-Box KD

## Brief Introduction

Knowledge distillation offers an effective solution by transferring knowledge from larger models to smaller ones, ensuring performance while significantly reducing computational resources and inference time. We introduce DistilQwen2, a lightweight LLM based on the Qwen2 series, optimized through enhanced instruction following and diverse distillation techniques. This enables more agile and efficient deployment in resource-constrained environments like mobile devices and edge computing. For ease of use by developers and enterprises, DistilQwen2's checkpoints are open-sourced on HuggingFace and ModelScope, empowering more stakeholders to innovate and realize value through advanced NLP applications.

## Instructional Data Processing Guidelines

For the training of DistilQwen2, we collected data from well-known open-source datasets like Magpie, Openhermes, and Mammoth 2, along with proprietary synthetic datasets to initiate the distillation process. The focus is on providing diverse instructional data, predominantly in Chinese and English. We also leverage prompt templates to conduct instructional data augmentation. Here, we provide several commonly used operations to re-sample and augement the dataset.

### Instruction Set Expansion

The instruction expansion operator is employed generate a diverse set of instruction variations, ensuring that student models are exposed to a comprehensive range of instructions. After instruction expansion, we can also call the teacher model to generate responses for new instructions. An example is calling this operator is as follows:

```bash
python easydistill/synthesis/synthesis_main.py --config=configs/instruction_expansion_api.json
```

If you need to run the job using batch inference, please refer to the config example `configs/instruction_expansion_batch.json`.

### Instruction Refinement

The instruction refinement operator further enhances the quality and diversity of the training data, which also preserves the semantic integrity of the tasks expressed in instructions, ensuring that the rewritten content remains faithful to the original intent and task category. After instruction refinement, we can also call the teacher model to generate responses for new instructions. An example is calling this operator is as follows:

```bash
python easydistill/synthesis/synthesis_main.py --config=configs/instruction_refinement_api.json
```

If you need to run the job using batch inference, please refer to the config example `configs/instruction_refinement_batch.json`.

### Instruction Resampling

We also consider task balance when selecting useful instructional data pairs. The task distrubutions are defined based on our paper in the reference. You can run the job by:

```bash
python task_resampling.py --input-file input.json --output-file output.json --api-key <your_api_key> --base-url <base_url>
```

The dataset is in JSON format, exemplified by entries such as:

```json
[
    {
        "instruction": "The ratio of the number of molar teeth in the human upper jaw at the age of 6 is 2:1 compared to number of incisors teeth. There are total 8 incisors in the human mouth..."
    },
    {
        "instruction": "This is the text of a lecture I am giving tomorrow. Can you go over it and make recommendations to improve clarity and flow?"
    }
]
```

After the processing of intructions, you can generate the responses of the teacher model.


### Open-Source Dataset

In addition, we have open-sourced part of the dataset used for model training, totaling 100K entries. This dataset includes mathematical problems, code tasks, Q&A, instruction following, and creative generation. Users can incorporate the DistilQwen_100K dataset, or its subsets, during model fine-tuning to enhance downstream task performance while maintaining generalization ability. The dataset is in JSON format, exemplified by entries such as:

```json
[
    {
        "instruction": "The ratio of the number of molar teeth in the human upper jaw at the age of 6 is 2:1 compared to number of incisors teeth. There are total 8 incisors in the human mouth...",
        "output": "## Step 1: Determine the total number of incisors in the upper jaw...\n\nThe final answer is: \\boxed{8}"
    },
    {
        "instruction": "This is the text of a lecture I am giving tomorrow. Can you go over it and make recommendations to improve clarity and flow?",
        "output": "I'd be happy to help you review your lecture text..."
    }
]
```

The dataset is available on ModelScope and Hugging Face. Users can download it using ModelScope's scripts and command-line tools.

```python
# Validate SDK token
from modelscope.hub.api import HubApi
api = HubApi()
api.login('your_token_id')

# Dataset download
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('PAI/DistilQwen_100k')
```

## Model Training Guidelines

### Black-Box KD

The black-box KD process follows a supervised learning paradigm, utilizing enhanced instruction-response pairs as training samples. Through this approach, the student model can effectively absorb and understand the knowledge imparted by the larger model, even with a limited number of parameters. This method not only boosts the student model's ability to tackle tasks but also enables it to perform better in multi-task scenarios. For simplicity, we use the `DistilQwen_100k` dataset as a tutorial, we need to run the training job only:

```bash
python easydistill/kd/train.py --config=distilqwen2_stage1.json
```

Plese refer to the config file `distilqwen2_stage1.json` in the current folder. If you need to run the job in a distributed mode, use `accelerate` to run the job.

### Preference Rank Optimization

For more challenging instruction tasks, SFT alone may not yield optimal results. To address this, we further refine the model using Direct Preference Optimization (DPO), enabling more granular fine-tuning and improved performance. Firstly, we generate the student outputs as rejected response. The contents in the SFT datasets are regarded as prompt and chosen responses. Please refer to the following script:

```bash
python dpo_student_infer_only.py --config=distilqwen2_stage2.json
```

Next, we run the training job by:

```bash
python easydistill/rank/train.py --config=distilqwen2_stage2.json
```

Again, please refer to the config file `distilqwen2_stage2.json` in the current folder. Remember to change the configurations when needed. If you need to run the job in a distributed mode, use `accelerate` to run the job.

## Model Download

We have open-sourced our distilled models on both HuggingFace and ModelScope. The available models are named `alibaba-pai/DistilQwen2-1.5B-Instruct` and `alibaba-pai/DistilQwen2-7B-Instruct`.

For example, users can download these models from HuggingFace using the following code:


```python
from huggingface_hub import snapshot_download

model_name = "alibaba-pai/DistilQwen2-1.5B-Instruct"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2-1.5B/")

model_name = "alibaba-pai/DistilQwen2-7B-Instruct"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2-7B/")
```


## Performance

The table below compares the performance of the original Qwen2 models with the distilled DistilQwen2 models across different parameter sizes: 1.5B and 7B. The evaluation metrics include AlpacaEval 2.0, MT-Bench, and IFEval scores. The distilled models demonstrate improved performance in instruction-following abilities over their respective original versions.

| Model                         | AlpacaEval 2.0 (length control) | MT-Bench         | MT-Bench (single) | IFEval (instruct-loose) | IFEval (strict-prompt) |
|-------------------------------|---------------------------------|------------------|-------------------|-------------------------|------------------------|
| Qwen2-1.5B-Instruct           | 5.22                            | 5.85             | 6.45              | 41.37                   | 28.10                  |
| **DistilQwen2-1.5B-Instruct** | **8.28**                        | **6.42**         | **7.12**          | **49.76**               | **36.04**              |
| Qwen2-7B-Instruct             | 24.33                           | 8.27             | 8.68              | 66.67                   | 52.31                  |
| **DistilQwen2-7B-Instruct**   | **25.35**                       | **8.40**         | **9.03**          | **71.46**               | **60.26**              |


	
## Reference

For more detailed information about the DistilQwen2 model series and the methodologies employed, we encourage you to refer to our paper:

- **Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning**  
  Yuanhao Yue, Chengyu Wang, Jun Huang, Peng Wang

You can cite the paper using the following citation format:

```bibtex
@inproceedings{emnlp2024,
  author       = {Yuanhao Yue and
                  Chengyu Wang and
                  Jun Huang and
                  Peng Wang},
  title        = {Distilling Instruction-following Abilities of Large Language Models with Task-aware Curriculum Planning},
  booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP} 2024},
  pages        = {6030--6054},
  publisher    = {Association for Computational Linguistics},
  year         = {2024},
  url          = {https://aclanthology.org/2024.findings-emnlp.350}
}
