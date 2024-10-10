# LoRA-Pro: Are Low-Rank Adapters Properly Optimized?

This repo contains the pre-release version of LoRA-Pro, proposed by [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/abs/2407.18242).

In LoRA-Pro, we uncover a fundamental connection between the optimization processes of LoRA and full fine-tuning: using LoRA for optimization is mathematically equivalent to full fine-tuning using a low-rank gradient for parameter updates. And this low-rank gradient can be expressed in terms of the gradients of the two low-rank matrices in LoRA. Leveraging this insight, we introduce LoRA-Pro, a method that enhances LoRA's performance by strategically adjusting the gradients of these low-rank matrices. This adjustment allows the low-rank gradient to more accurately approximate the full fine-tuning gradient, thereby narrowing the performance gap between LoRA and full fine-tuning. Furthermore, we theoretically derive the optimal solutions for adjusting the gradients of the low-rank matrices, applying them during fine-tuning in LoRA-Pro.



## Requirements

### Installation

Create a conda environment and install dependencies:

```shell
git clone https://github.com/mrflogs/LoRA-Pro.git
cd LoRA-Pro

conda activate -n lorapro python=3.9
conda activate lorapro

# install required package
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
```

### Datasets and Models

Install Llama-2-7B from huggingface and link it to `./models`.

Install datasets (WizardLM, MetaMathQA, and CodeFeedback-Filtered-Instruction, etc.) from huggingface and link them to `./data`.

## Get Started

### Example

#### 0. Configure custom trainer with lora-pro optimizer

```python
import peft
import peta
from transformers import TrainingArguments, Trainer 

# Configure custom trainer with lora-pro optimizer
class CustomTrainer(Trainer):
    def __init__(self, *args, scaling_factor=2, **kwargs):
        super().__init__(*args, **kwargs)

        self.scaling_factor = scaling_factor
        self.lora_modules = []
        self.find_modules(self.model, self.lora_modules)

    def find_modules(self, module ,lora_modules):
        for sub_module in module.children():
            if isinstance(sub_module, peft.tuners.lora.layer.Linear):
                lora_modules.append(sub_module)
            elif list(sub_module.children()):
                self.find_modules(sub_module, lora_modules)
        
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = peta.optim.AdamW(self.model.named_parameters(), lr=self.args.learning_rate, scaling_factor=self.scaling_factor, betas=(0.9, 0.999), weight_decay=self.args.weight_decay, mode="efficient", X_mode='sylvester')
        
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)

```

#### 1. Training 

```
# LoRA-Pro Trainer
trainer = CustomTrainer(
    scaling_factor=(lora_config.lora_alpha / lora_config.r),
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["eval"],
    tokenizer=tokenizer,
    args=train_args,
    data_collator=default_data_collator,
)

# Train
trainer.train()
```

### Reproduce Llama-2-7b results

#### 0. Training

The training scripts  can be found in `./scripts/llama-2-7b_transformers.sh`

For math task,

```shell
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora rslora-pro --seed 0
```

For code task,

```shell
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora rslora-pro --seed 0
```

For chat task,

```shell
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora rslora-pro --seed 0
```

#### 1. Evaluation

For math task,

```shell
torchrun --nproc_per_node=8 evaluation/eval_llama-2_math_multi_gpus.py
```

For code task, we generate results with script below and evaluate its PASS@1 using [HumanEval](https://github.com/openai/human-eval), 

```shell
torchrun --nproc_per_node=8 evaluation/eval_llama-2_code_multi_gpus.py
```

For chat task, we use [FastChat](https://github.com/lm-sys/FastChat/tree/main) to generation and evaluate with GPT-4, please read their instruction.



## Citation
```latex
@article{wang2024lorapro,
  title={LoRA-Pro: Are Low-Rank Adapters Properly Optimized?},
  author={Wang, Zhengbo and Liang, Jian and He, Ran and Wang, Zilei and Tan, Tieniu},
  journal={arXiv preprint arXiv:2407.18242},
  year={2024}
}
```

## Contact

If you have any question, feel free to contact 📫zhengbowang@mail.ustc.edu.cn.









