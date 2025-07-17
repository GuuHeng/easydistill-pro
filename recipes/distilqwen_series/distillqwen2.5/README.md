# DistilQwen2.5: Combining Black-Box and White Box KD

## Brief Introduction

The DistilQwen2.5 distilled language model series is built upon the Qwen2.5 model. This series leverages innovative distillation techniques to enhance instruction-following capabilities. As a result, these distilled models retain the excellent performance of the original models while requiring fewer computational resources.

The distillation process involves carefully selecting, rewriting, and optimizing instruction-response pairs conducive to student model learning, thus improving model comprehension and execution abilities. Following standard fine-tuning, we employ white-box distillation techniques to enable the student models to better acquire fine-grained knowledge from teacher models. Experimental evaluations demonstrate the significant improvement in capabilities of the DistilQwen2.5 models. 

## Detailed Steps

### Processing of Instructional Dataset

DistilQwen2.5 begins with collecting diverse, high-quality instructional data from sources like Magpie, Openhermes, and Mammoth 2, along with proprietary datasets. This data includes Chinese and English instructions, scoring them for difficulty and task relevance. This process is very similar to the recipe of DistilQwen2.

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

### Black-Box KD

The black-box KD process follows a supervised learning paradigm, utilizing enhanced instruction-response pairs as training samples. Through this approach, the student model can effectively absorb and understand the knowledge imparted by the larger model, even with a limited number of parameters. This method not only boosts the student model's ability to tackle tasks but also enables it to perform better in multi-task scenarios. Because we have already obtained the teacher's responses in the dataset, we need to run the training job only:

```bash
python easydistill/kd/train.py --config=distilqwen2.5_stage1.json
```

Plese refer to the config file `distilqwen2.5_stage1.json` in the current folder. If you need to run the job in a distributed mode, use `accelerate` to run the job.

### White-Box KD

Unlike black-box KD, which relies solely on the highest probability token output by the teacher model, white-box KD focuses on the distribution of logits produced by the teacher model. This approach provides the student model with richer information. By mimicking the teacher model's logits distribution, white-box KD can transfer knowledge more effectively, thereby enhancing the performance of the student model. As an example, we take `Qwen2.5-72B-Instruct` as the white-box teacher model, and generate the logits by:

```bash
python easydistill/kd/infer.py --config=distilqwen2.5_stage2.json
```

Next, we run the training job by:

```bash
python easydistill/kd/train.py --config=distilqwen2.5_stage2.json
```

Again, please refer to the config file `distilqwen2.5_stage2.json` in the current folder. Remember to change the configurations when needed.

## Model Download

We have open-sourced our distilled models on both HuggingFace and ModelScope. The available models are named `alibaba-pai/DistilQwen2.5-0.5B-Instruct`, `alibaba-pai/DistilQwen2.5-1.5B-Instruct`, `alibaba-pai/DistilQwen2.5-3B-Instruct`, and `alibaba-pai/DistilQwen2.5-7B-Instruct`.

For example, users can download these models from HuggingFace using the following code:


```python
from huggingface_hub import snapshot_download

# Download the 0.5B model
model_name = "alibaba-pai/DistilQwen2.5-0.5B-Instruct"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-0.5B/")

# Download the 1.5B model
model_name = "alibaba-pai/DistilQwen2.5-1.5B-Instruct"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-1.5B/")

# Download the 3B model
model_name = "alibaba-pai/DistilQwen2.5-3B-Instruct"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-3B/")

# Download the 7B model
model_name = "alibaba-pai/DistilQwen2.5-7B-Instruct"
snapshot_download(repo_id=model_name, cache_dir="./DistilQwen2.5-7B/")
```


## Performance

The table below compares the performance of the original Qwen2.5 models with the distilled DistilQwen2.5 models across different parameter sizes: 0.5B, 1.5B, 3B, and 7B. The evaluation metrics include AlpacaEval 2.0, MT-Bench, and IFEval scores. The distilled models demonstrate improved performance in instruction-following abilities over their respective original versions.

| Model                         | AlpacaEval 2.0 (length control) | MT-Bench         | MT-Bench (single) | IFEval (instruct-loose) | IFEval (strict-prompt) |
|-------------------------------|---------------------------------|------------------|-------------------|-------------------------|------------------------|
| Qwen2.5-0.5B-Instruct         | 2.46                            | 5.49             | 6.26              | 42.81                   | 30.31                  |
| **DistilQwen2.5-0.5B-Instruct** | **4.89**                        | **5.78**         | **6.83**          | **52.61**               | **37.82**              |
| Qwen2.5-1.5B-Instruct         | 6.69                            | 7.09             | 7.66              | 55.40                   | 40.11                  |
| **DistilQwen2.5-1.5B-Instruct** | **13.69**                       | **7.35**         | **7.99**          | **61.10**               | **74.49**              |
| Qwen2.5-3B-Instruct           | 17.98                           | 7.92             | 8.40              | 61.18                   | 74.58                  |
| **DistilQwen2.5-3B-Instruct**   | **20.91**                       | **8.37**         | **8.97**          | **67.03**               | **77.36**              |
| Qwen2.5-7B-Instruct           | 31.43                           | 8.52             | 8.83              | 81.53                   | 72.10                  |
| **DistilQwen2.5-7B-Instruct**   | **34.86**                       | **8.76**         | **9.22**          | **83.48**               | **73.27**              |


For evaluation details, please refer to our paper.

## Reference

For more detailed information about the DistilQwen2.5 model series and the methodologies employed, we encourage you to refer to our paper:

- **DistilQwen2.5: Industrial Practices of Training Distilled Open Lightweight Language Models**  
  Chengyu Wang, Junbing Yan, Yuanhao Yue, Jun Huang  
  [arXiv:2504.15027](https://arxiv.org/abs/2504.15027)

You can cite the paper using the following citation format:

```bibtex
@misc{wang2025distilqwen25industrialpracticestraining,
      title={DistilQwen2.5: Industrial Practices of Training Distilled Open Lightweight Language Models}, 
      author={Chengyu Wang and Junbing Yan and Yuanhao Yue and Jun Huang},
      year={2025},
      eprint={2504.15027},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.15027}, 
}
```