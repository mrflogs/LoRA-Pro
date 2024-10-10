# Math
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora rslora-pro --seed 0
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora rslora-pro --seed 1
torchrun --nproc_per_node=8 minimal_lora_llama2_math_transformers.py --lora rslora-pro --seed 2

# Code 
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora rslora-pro --seed 0
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora rslora-pro --seed 1
torchrun --nproc_per_node=8 minimal_lora_llama2_code_transformers.py --lora rslora-pro --seed 2

# Chat 
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora rslora-pro --seed 0
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora rslora-pro --seed 1
torchrun --nproc_per_node=8 minimal_lora_llama2_chat_transformers.py --lora rslora-pro --seed 2














