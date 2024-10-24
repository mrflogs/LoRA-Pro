from datasets import load_dataset, Dataset, DatasetDict
import typing as tp
import functools
import os
import pickle
from .datasets_preprocess import DatasetPreprocessor, preprocess

def cache_to_disk(root_datadir):
    def decorator_cache(func):
        @functools.wraps(func)
        def wrapper_cache(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_file = os.path.join(root_datadir, f"{func_name}.pkl")

            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper_cache

    return decorator_cache


class WizardLM52k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["instruction"], str):
            raise NotImplementedError
    
        else:
            combined_text = [
                x + " " + y + self.tokenizer.eos_token 
                for (x, y) in zip(example["instruction"], example["output"])
            ]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding="max_length", truncation=True, max_length=1024)
            input_text_length = [
                len(self.tokenizer(example["instruction"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["instruction"]))
            ]
            labels = encodings["input_ids"].clone()
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

template_wo_input = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
'''
        
@cache_to_disk("data_cache")
def load_wizardlm(max_tokens=1024):
        
    dataset = load_dataset("./data/Wizard-LM-Chinese-instruct-evol", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/llama-2/llama-2-7b")
    def preprocess(data):
        y = data['output']
        return {
            "instruction": template_wo_input.format(
                instruction=data['instruction']
            ),
            "output": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=70000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "sorry" in temp['output'].lower() or "as an ai" in temp['output'].lower():
            continue
        if len(tokenizer(temp['instruction']+' '+temp['output'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = temp
        if count < 52000:
            train_samples.append(processed_sample)
        elif 52000 <= count < 70000:
            eval_samples.append(processed_sample)
        elif count >= 70000:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })
    
    return datasets


class MetaMathQA100k_Preprocessor(DatasetPreprocessor):
    # [TODO]

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["x"], str):
            # not batched
#             input_text, target_text = self.preprocess(
#                 example["instruction"], example["output"]
#             )
            raise NotImplementedError
    
        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

            labels = encodings["input_ids"].clone()
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
#                 "attention_mask": encodings["input_ids"].ne(self.tokenizer.pad_token_id),
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

@cache_to_disk("data_cache")
def load_meta_math(max_tokens=512):
    
    dataset = load_dataset("./data/MetaMathQA", split="train")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/llama-2/llama-2-7b")
#     def preprocess(data):
#         return {
#             "x": f'Q: {data["query"]}\nA: ',
#             "y": data["response"].split("\nThe answer is:")[0]
#         }
    def preprocess(data):
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": data["response"]
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens or "GSM" not in sample["type"]:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:  # First 100,000 samples for training
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:  # Next 10,000 samples for evaluation
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1

    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })
    
    return datasets

@cache_to_disk("data_cache")
def load_gsm8k():
    dataset = load_dataset("gsm8k", "main")
    #x = "Q: " + x[0] + "\n" + "A:"
    dataset = dataset.map(
        lambda e: {
            "x": f'Q: {e["question"]}\nA: ',
            "y": e["answer"],
        }
    )
    train_set = dataset["train"]
    validation_set = dataset["test"]
    return train_set, validation_set, validation_set


class CodeFeedback100k_Preprocessor(DatasetPreprocessor):

    def __call__(self, example):
        """
        Preprocess the CoLA dataset into a text-to-text format.
        """
        if isinstance(example["x"], str):
            # not batched
            raise NotImplementedError
    
        else:
            combined_text = [(x + " " + y + self.tokenizer.eos_token) for (x, y) in zip(example["x"], example["y"])]
            encodings = self.tokenizer(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=1024)

            labels = encodings["input_ids"].clone()
            input_text_length = [
                len(self.tokenizer(example["x"][i], return_tensors="pt")["input_ids"][0])
                for i in range(len(example["x"]))
            ]
            for i, l in enumerate(input_text_length):
                labels[i, :l] = -100
            labels[encodings["attention_mask"] == 0] = -100
            
            results = {
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": labels,
            }

            return results

@cache_to_disk("data_cache")
def load_codefeedback(max_tokens=1024):
    dataset = load_dataset("./data/CodeFeedback-Filtered-Instruction", split='train')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("./models/llama-2/llama-2-7b")
    def preprocess(data):
        y = data['answer']
        y = "```".join(y.split("```")[:2]) + "```" # only keep the first code block
        return {
            "x": template_wo_input.format(
                instruction=data['query']
            ),
            "y": y,
        }
    train_samples = []
    eval_samples = []
    count = 0
    dataset.shuffle(seed=42)
    from tqdm import tqdm
    bar = tqdm(dataset, total=110000)
    total = 0
    ok = 0
    for sample in dataset:
        total += 1
        temp = preprocess(sample)
        if "```" not in sample['answer']:
            continue
        if len(tokenizer(temp['x']+' '+temp['y'])['input_ids']) >= max_tokens:
            continue
        bar.update(1)
        bar.set_description(f"ok: {ok}/{total}")
        ok += 1
        processed_sample = preprocess(sample)
        if count < 100000:
            train_samples.append(processed_sample)
        elif 100000 <= count < 110000:
            eval_samples.append(processed_sample)
        elif count >= 110000:  # Stop processing after collecting enough samples
            break
        count += 1
        
    # convert to hf dataset
    train_samples = Dataset.from_list(train_samples)
    eval_samples = Dataset.from_list(eval_samples)
    datasets = DatasetDict({
        "train": train_samples,
        "eval": eval_samples,
    })
    
    return datasets
        
#     # convert to hf dataset
#     train_set = Dataset.from_list(train_samples)
#     eval_set = Dataset.from_list(eval_samples)
#     return train_set, eval_set, eval_set





























































