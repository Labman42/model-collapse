import math
import os
import argparse
import json
from datasets import concatenate_datasets, DatasetDict
from dataset import prepare_data, preprocess_datasets, preprocess_datasets_ungroup
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

def evaluate_model(trainer, dataset, description):
    results = trainer.evaluate(eval_dataset=dataset)
    loss = results['eval_loss']
    perplexity = math.exp(loss)
    print(f"{description} loss: {loss:.4f}, perplexity: {perplexity:.2f}")
    return loss, perplexity

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the opt with dirty data.")
    parser.add_argument("old", type=str, help="Path to the directory containing the checkpoint folder")
    parser.add_argument("directory", type=str, help="Path to the directory containing the checkpoint folder")
    args = parser.parse_args()


    OLD_VERSION = args.old
    VERSION = args.directory
    CACHE_DIR = '/fs/class-projects/fall2024/cmsc473/c473g001/'

    # Step 1: Load Tokenizer and Model
    checkpoints_dir = f"/fs/class-projects/fall2024/cmsc473/c473g001/base/{OLD_VERSION}"

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


    # Load and preprocess the data
    raw_dataset = prepare_data(path=f"./generated_dataset_{OLD_VERSION}")
    tokenized_datasets = preprocess_datasets(raw_dataset, tokenizer)
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets['validation']
    test_dataset = tokenized_datasets['test']

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Step 3: Fine-tune the Model
    training_args = TrainingArguments(
        output_dir=f"/fs/class-projects/fall2024/cmsc473/c473g001/base/{VERSION}",
        overwrite_output_dir=True,
        num_train_epochs=5,  # Adjust as needed
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Start Training!!")
    trainer.train()

    print("Start Evaluation!!")
    train_loss, train_perplexity = evaluate_model(trainer, train_dataset, 'Train')
    val_loss, val_perplexity = evaluate_model(trainer, val_dataset, 'Validation')
    test_loss, test_perplexity = evaluate_model(trainer, test_dataset, 'Test')

    # Organizing the data into a dictionary
    results = {
        "Train": {"loss": train_loss, "perplexity": train_perplexity},
        "Validation": {"loss": val_loss, "perplexity": val_perplexity},
        "Test": {"loss": test_loss, "perplexity": test_perplexity}
    }

    # Saving to a JSON file
    file_path = f"evaluation_results_{VERSION}.json"  # Update with your desired path
    with open(file_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {file_path}")