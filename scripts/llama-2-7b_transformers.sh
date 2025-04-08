# Math
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora lora-pro-full --seed 0 --lr 0.00002
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora lora-pro-full --seed 1 --lr 0.00002
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora lora-pro-full --seed 2 --lr 0.00002

# Code 
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora lora-pro-full --seed 0 --lr 0.00002
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora lora-pro-full --seed 1 --lr 0.00002
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora lora-pro-full --seed 2 --lr 0.00002

# Chat 
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora lora-pro-full --seed 0 --lr 0.00002
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora lora-pro-full --seed 1 --lr 0.00002
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora lora-pro-full --seed 2 --lr 0.00002














