import os
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from itertools import chain
from transformers import DataCollatorForLanguageModeling


def prepare_data(path='./data/wikitext2'):
    if (path is not None) and (not os.path.isdir(path)):
        print("Downloading and processing dataset...")
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        dataset.save_to_disk(path)
    else:
        print("Dataset already downloaded and processed")
        dataset = load_from_disk(path)
    return dataset


def preprocess_datasets(
        raw_dataset, 
        tokenizer, 
        block_size=64, 
        overwrite_cache=False, 
        preprocessing_num_workers=4):
    column_names = raw_dataset['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if block_size is None:
        block_size = tokenizer.model_max_length

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not overwrite_cache,
        desc="Running tokenizer on dataset",
        keep_in_memory=True,
    )

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len (block_size).
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=preprocessing_num_workers,
        load_from_cache_file=not overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
        keep_in_memory=True
    )
    return dataset

def get_dataloader(tokenizer, context_len, batch_size=128, shuffle=True):
    raw_dataset = prepare_data()
    dataset = preprocess_datasets(
        raw_dataset, tokenizer, block_size=context_len)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return train_dataloader, val_dataloader, test_dataloader
