import os
import argparse
import torch
from datasets import concatenate_datasets, DatasetDict
from dataset import prepare_data, preprocess_datasets_ungroup
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

def find_single_checkpoint(directory):
    # List all entries in the directory
    entries = os.listdir(directory)
    
    # Filter for directories only
    folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
    
    # Check if there's exactly one folder
    if len(folders) == 1:
        return os.path.join(directory, folders[0])
    else:
        raise Exception(f"Expected exactly one folder in {directory}, but found {len(folders)}.")

def generate_batch_texts(examples):
    input_ids = examples['input_ids'][:64]
    input_ids = tokenizer.pad(
        {'input_ids': input_ids},
        padding='longest',
        return_tensors='pt'
    )['input_ids'].to(device)
    
    # Generate outputs
    outputs = model.module.generate(  # Use `module` when using DataParallel
        input_ids,
        max_new_tokens=512,
        min_new_tokens=64,
        repetition_penalty=3.0,
    )

    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return {'text': generated_texts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data with trained opt.")
    parser.add_argument("old", type=str, help="Path to the directory containing the checkpoint folder")
    parser.add_argument("new", type=str, help="Path to the directory containing the checkpoint folder")
    args = parser.parse_args()

    OLD_VERSION = args.old
    NEW_VERSION = args.new
    # Path to the directory containing the checkpoint folder
    checkpoints_dir = f"/fs/class-projects/fall2024/cmsc473/c473g001/results/{OLD_VERSION}"

    # Find the checkpoint folder
    try:
        checkpoint_path = find_single_checkpoint(checkpoints_dir)
        print(f"The checkpoint folder is: {checkpoint_path}")
    except Exception as e:
        print(str(e))

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/opt-125m", 
        cache_dir='/fs/class-projects/fall2024/cmsc473/c473g001/cache/'
    )
    tokenizer.padding_side = "left"  # Ensure left-padding for decoder-only models
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        # torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2"
    )

    # Move model to GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Load and preprocess the data
    raw_dataset = prepare_data()
    tokenized_datasets = preprocess_datasets_ungroup(raw_dataset, tokenizer)
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    test_dataset = tokenized_datasets['test']

    print("Start Generating!!")
    generate_percentage = 1  # Adjust percentage as needed
    if generate_percentage > 0:
        num_examples = int(len(train_dataset) * generate_percentage)
        
        generated_subset = train_dataset.select(range(len(train_dataset) - num_examples, len(train_dataset)))
        
        generated_subset = generated_subset.map(generate_batch_texts, batched=True, batch_size=16)
        
        generated_subset = generated_subset.map(lambda x: {'text': x['text']}, remove_columns=generated_subset.column_names)

    # Get the unused subset and retain only the 'text' column
    unused_subset = train_dataset.select(range(0, len(train_dataset) - num_examples))
    unused_subset = unused_subset.map(
        lambda x: {'text': tokenizer.decode(x['input_ids'], skip_special_tokens=True)},
        remove_columns=unused_subset.column_names
    )

    # Combine generated and unused subsets into a new dataset with only the 'text' feature
    combined_text_dataset = concatenate_datasets([unused_subset, generated_subset]) if generate_percentage > 0 else unused_subset

    # Combine with validation and test datasets to create a DatasetDict
    dataset_dict = DatasetDict({
        'train': combined_text_dataset, 
        'validation': raw_dataset['validation'], 
        'test': raw_dataset['test']
    })

    # Step 6: Save the Combined Dataset
    dataset_dict.save_to_disk(f"generated_dataset_{NEW_VERSION}")