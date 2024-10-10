import os
import argparse
import logging
from pathlib import Path

import lightning as L

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from datasets import DatasetDict, load_dataset
import transformers
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator

import peft
from peft import LoraConfig

import peta
from peta.utils import TitledLog

import wandb

import math

log = logging.getLogger(__name__)

# disable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"


torch.set_float32_matmul_precision("high")

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
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora', default="lora-pro", type=str)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()
    
    assert args.lora in ["lora-pro", "rslora-pro"]
    
    return args        

def main():
    
    lora_r = 8
    lora_alpha = 16
    
    args = get_arguments()
    
    log.info(f"set seed to {args.seed}")
    L.seed_everything(args.seed)
    set_seed(args.seed)
    
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    
    # Initialize Wandb
    if local_rank == 0:
        wandb.init(
            project='LLAMA-2-7B',  
            name=f"llama-2-7b_code_rank_{lora_r}_{args.lora}_seed_{args.seed}",
            group='Transformers-Code',
        )
    
    # Step 0: load model and tokenizer
    path = "./models/llama-2/llama-2-7b"
    tokenizer = transformers.LlamaTokenizer.from_pretrained(path)
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        model.resize_token_embeddings(len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = transformers.LlamaForCausalLM.from_pretrained(path, max_length=1024, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map={"": int(os.environ.get("LOCAL_RANK") or 0)})

    # Step 1: Peft model
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        use_rslora=True if "rs" in args.lora else False,
    )
    
    model = peft.get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    for name, module in model.named_modules():
        if "lora_" in name:  
            module.to(torch.float32)

    # Step 2: load dataset
    with TitledLog("load datasets and dataloaders", log_fn=log.info):
        datasets = peta.tasks.load_codefeedback()
        
        preprocessor = peta.tasks.CodeFeedback100k_Preprocessor(
            tokenizer=tokenizer,
            tokenizer_kwargs={
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "pt",
                "max_length": 1024
            },
        )

        datasets = datasets.map(
            preprocessor,
            batched=True,
            batch_size=1000,
            num_proc=1,
            desc="Running tokenizer on dataset",
        )
        
    # Step 3: Training Args
    train_args = TrainingArguments(
        output_dir="./llama-2-7b-codefeedback100k",
        logging_dir="./transformers_logs",
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        logging_steps=1,
        bf16=True,
        learning_rate=2e-5,
        weight_decay=0, # No weight decay
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="wandb" if local_rank == 0 else None, 
        label_names=[
            "labels"
        ],  
        ddp_find_unused_parameters=False,
        do_eval=True,
        per_device_eval_batch_size=1,
        evaluation_strategy="steps",
        eval_steps=0.1,
        save_strategy="no",
        deepspeed="./config/deepspeed_zero2.json" if world_size > 1 else None, 
    )    
    
    # Step 4: Trainer
    trainer = CustomTrainer(
        scaling_factor=(lora_config.lora_alpha / math.sqrt(lora_config.r)) if "rs" in args.lora else (lora_config.lora_alpha / lora_config.r),
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        tokenizer=tokenizer,
        args=train_args,
        data_collator=default_data_collator,
    )
    
    # Step 5: Train
    trainer.train()
    
    # Step 6: Save model
    if local_rank == 0:
        model.save_pretrained(f'./logs/transformers/llama-2-7b/code/{args.lora}/{args.seed}')
        tokenizer.save_pretrained(f'./logs/transformers/llama-2-7b/code/{args.lora}/{args.seed}')

        wandb.finish()
            

if __name__ == "__main__":
    main()









































