import os
import math
from itertools import chain
from datasets import load_dataset, load_from_disk
from dataset import prepare_data, preprocess_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# Step 1: Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m", cache_dir='/fs/class-projects/fall2024/cmsc473/c473g001/cache/')
# model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir='/fs/class-projects/fall2024/cmsc473/c473g001/cache/')


# Load and preprocess the data
# raw_dataset = prepare_data()
# print(raw_dataset)
# tokenized_datasets = preprocess_datasets(raw_dataset, tokenizer, block_size=64)
# train_dataset = tokenized_datasets['train']
# print(train_dataset)


raw_dataset = prepare_data(path='./generated_dataset')
tokenized_datasets = preprocess_datasets(raw_dataset, tokenizer, block_size=64)