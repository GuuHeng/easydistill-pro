# DistilQwen-100k/DistilQwen-1M: High-Quality Instruction-Tuning Datasets  

## Overview  
To empower community developers in enhancing the **instruction-following capabilities** of large language models (LLMs), we open-source **`DistilQwen-100k`** and **`DistilQwen-1M`**, subsets of the training data used for the **DistilQwen model series**. The datasets provide diverse, high-quality samples to improve model performance in key areas.  

## Dataset Features  
- **Scale**: **100 thousand**/**1 million** meticulously distilled entries.  
- **Coverage**: Balanced mix of:  
  - **Mathematics**  
  - **Code generation & understanding**  
  - **Knowledge-based QA**  
  - **Instruction following**  
  - **Creative generation**  
- **Purpose**: Optimized for **instruction tuning**, helping models retain generalization while adapting to downstream tasks.  

## Use Cases  
- **Fine-tuning LLMs**: Mitigate *catastrophic forgetting* by combining with custom datasets.  
- **Multi-task learning**: Improve coherence in mathematical reasoning, coding, and creative tasks.  
- **Research**: Study distillation techniques or instruction-tuning efficacy.  

## Use the Datasets
```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("alibaba-pai/DistilQwen_100k")
ds = load_dataset("alibaba-pai/DistilQwen_1M")
```

## Reference

For more detailed information about the dataset construction process, we encourage you to refer to our paper:

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
      url={https://arxiv.org/abs/2504.15027}
}
```