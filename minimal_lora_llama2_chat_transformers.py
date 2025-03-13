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
    parser.add_argument('--lr', default=2e-5, type=float)
    args = parser.parse_args()

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
            name=f"llama-2-7b_chat_rank_{lora_r}_{args.lora}_seed_{args.seed}",
            group='Transformers-Chat',
        )
    
    # Step 0: load model and tokenizer
    path = "./models/llama-2-7b"
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
        lora_dropout=0.,
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
        datasets = peta.tasks.load_wizardlm()
        
        preprocessor = peta.tasks.WizardLM52k_Preprocessor(
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
    gradient_accumulation_steps = 32 // world_size
    train_args = TrainingArguments(
        output_dir="./llama-2-7b-wizardlm52k",
        logging_dir="./transformers_logs",
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="sgd",
        logging_steps=1,
        bf16=True,
        learning_rate=2e-5,
        weight_decay=0., # No weight decay
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
        eval_steps=-1,
        save_strategy="no",
        deepspeed="./config/deepspeed_zero2.json" if world_size > 1 else None, 
    )   
    
    # Step 4: Trainer
    trainer = Trainer(
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
        model.save_pretrained(f'./logs/transformers/llama-2-7b/chat/{args.lr}/{args.lora}/{args.seed}')
        tokenizer.save_pretrained(f'./logs/transformers/llama-2-7b/chat/{args.lr}/{args.lora}/{args.seed}')
        print(f"Saving at path: ./logs/transformers/llama-2-7b/chat/{args.lr}/{args.lora}/{args.seed}")
        wandb.finish()
            

if __name__ == "__main__":
    main()









































