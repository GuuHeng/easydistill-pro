# OmniThought: A Large-Scale Chain-of-Thought Dataset for Advancing Large Reasoning Models  

## Overview  
The rise of **Large Reasoning Models (LRMs)** has revolutionized **Natural Language Processing (NLP)**, enabling breakthroughs in complex tasks like **mathematical problem-solving** and **code generation**. These models rely on **Chain-of-Thought (CoT)** processes to mimic human-like reasoning. However, progress in LRMs is limited by the scarcity of **high-quality, large-scale CoT datasets**—existing resources often lack:  
- **Diverse reasoning problems** with well-structured CoT processes.  
- **Multi-teacher distillation** to ensure reasoning quality.  
- **Fine-grained annotations** describing CoT properties.  

To bridge this gap, we introduce **`OmniThought`**, a **2-million-scale CoT dataset** generated and validated by **two powerful LRMs**. Each CoT process is annotated with:  
- **Reasoning Verbosity (RV)**: Measures the optimal verbosity of reasoning steps.  
- **Cognitive Difficulty (CD)**: Assesses the complexity of reasoning for model comprehension.  

We also propose a **self-reliant pipeline** for dataset curation, ensuring high-quality reasoning traces.  

## Key Features  
✅ **2 million high-quality CoT processes** covering diverse reasoning tasks.  
✅ **RV-CD scores** to guide model training for better reasoning performance.  
✅ **Multi-teacher distillation** for robust and coherent reasoning paths.  
✅ **Optimized for LRM training**—improves reasoning ability and output quality.  

## Experiments & Results  
Extensive experiments with **Qwen2.5 models** (various sizes) confirm that:  
- Training with **RV-CD scores** enhances **LRM reasoning effectiveness**.  
- Models trained on `OmniThought` achieve **stronger reasoning abilities** with **optimal CoT length and difficulty**.  

Based on this dataset, we release **a series of high-performance LRMs** with superior reasoning capabilities.  

## Use the Datasets
```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("alibaba-pai/OmniThought")
```



## Reference

For more detailed information about the dataset construction process, we encourage you to refer to our paper:

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