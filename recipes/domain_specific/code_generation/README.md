# Code Generation: Combining Black-Box and White Box KD

## Brief Introduction

The code generation distillation process involves carefully selecting and optimizing code generation instruction-response pairs conducive to student model learning. Following standard fine-tuning, we employ white-box distillation techniques to enable the student models to better acquire fine-grained knowledge from teacher models.

## Detailed Steps

### Processing of Code Generation Dataset

The process begins with collecting diverse, high-quality code generation data from various programming languages and tasks. This data includes code completion, code generation from natural language descriptions, bug fixing, and code explanation tasks. We score examples for difficulty and relevance to code generation tasks.

The dataset is in JSON format, exemplified by entries such as:

```json
[
    {
        "instruction": "Write a Python function that calculates the Fibonacci sequence up to n elements using dynamic programming.",
        "output": "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[i-1] + fib[i-2])\n    \n    return fib\n\n# Example usage\nprint(fibonacci(10))  # Output: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"
    },
    {
        "instruction": "Create a JavaScript function that takes an array of numbers and returns the sum of all even numbers in the array.",
        "output": "function sumEvenNumbers(numbers) {\n  return numbers\n    .filter(num => num % 2 === 0)\n    .reduce((sum, num) => sum + num, 0);\n}\n\n// Example usage\nconst numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];\nconsole.log(sumEvenNumbers(numbers));  // Output: 30"
    }
]
```

### Black-Box KD

The black-box KD process follows a supervised learning paradigm, utilizing enhanced code instruction-response pairs as training samples. Through this approach, the student model can effectively absorb and understand the code generation capabilities of the larger model, even with a limited number of parameters. This method not only boosts the student model's ability to tackle programming tasks but also enables it to perform better across multiple programming languages and paradigms.

To run the black-box KD training:

```bash
python easydistill/kd/train.py --config=code_generation_stage1.json
```

Please refer to the config file `code_generation_stage1.json` in the current folder. If you need to run the job in a distributed mode, use `accelerate` to run the job.

### White-Box KD

Unlike black-box KD, which relies solely on the highest probability token output by the teacher model, white-box KD focuses on the distribution of logits produced by the teacher model. This approach provides the student model with richer information about code structure and syntax. By mimicking the teacher model's logits distribution, white-box KD can transfer programming knowledge more effectively, thereby enhancing the performance of the student model.

To generate the logits with the teacher model:

```bash
python easydistill/kd/infer.py --config=code_generation_stage2.json
```

Next, run the training job:

```bash
python easydistill/kd/train.py --config=code_generation_stage2.json
```

Please refer to the config file `code_generation_stage2.json` in the current folder. Remember to change the configurations when needed. 

## Performance

We trained the model using data from nvidia/OpenCodeReasoning, and the final model performance is as follows:

| Model                     | LiveCodeBench V2 | speed  |
|---------------------------|------------------|--------|
| Qwen2.5-3B-Instruct       | 11.35            | 2.3x   |
| Qwen2.5-3B-Code-Optimize  | 16.62            | 2.3x   |
| Qwen2.5-7B-Instruct       | 30.72            | 1x     |
| Qwen2.5-7B-Code-Optimize  | 35.32            | 1x     |