import os
import math
from itertools import chain
from datasets import concatenate_datasets, DatasetDict
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
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", cache_dir='/fs/class-projects/fall2024/cmsc473/c473g001/cache/')


# Load and preprocess the data
raw_dataset = prepare_data()
tokenized_datasets = preprocess_datasets(raw_dataset, tokenizer, block_size=64)
train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['validation']
test_dataset = tokenized_datasets['test']

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 3: Fine-tune the Model
training_args = TrainingArguments(
    output_dir='/fs/class-projects/fall2024/cmsc473/c473g001/results',
    overwrite_output_dir=True,
    num_train_epochs=5,  # Adjust as needed
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
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
# trainer.train()

# Step 4: Evaluate the Model
def evaluate_model(trainer, dataset, description):
    results = trainer.evaluate(eval_dataset=dataset)
    loss = results['eval_loss']
    perplexity = math.exp(loss)
    print(f"{description} loss: {loss:.4f}, perplexity: {perplexity:.2f}")
    return loss, perplexity

print("Start Evaluation!!")
train_loss, train_perplexity = evaluate_model(trainer, train_dataset, 'Train')
val_loss, val_perplexity = evaluate_model(trainer, val_dataset, 'Validation')
test_loss, test_perplexity = evaluate_model(trainer, test_dataset, 'Test')


def generate_batch_texts(examples):
    input_ids = examples['input_ids']
    input_ids = tokenizer.pad(
        {'input_ids': input_ids},
        padding='longest',
        return_tensors='pt'
    )['input_ids'].to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=64,
        num_beams=5,
        repetition_penalty=3.0,
    )
    generated_ids = outputs[:, input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {'text': generated_texts}


print("Start Generating!!")
# Step 5: Generate New Data
generate_percentage = 0.3  # Adjust percentage as needed
if generate_percentage > 0:
    num_examples = int(len(train_dataset) * generate_percentage)
    
    # Select subset for generation
    generated_subset = train_dataset.select(range(len(train_dataset) - num_examples, len(train_dataset)))
    
    # Generate new texts
    generated_subset = generated_subset.map(generate_batch_texts, batched=True, batch_size=128)
    
    # Keep only the 'text' column
    generated_subset = generated_subset.map(lambda x: {'text': x['text']}, remove_columns=generated_subset.column_names)

# Get the unused subset and retain only the 'text' column
unused_subset = train_dataset.select(range(0, len(train_dataset) - num_examples))
unused_subset = unused_subset.map(
    lambda x: {'text': tokenizer.decode(x['input_ids'], skip_special_tokens=True)},
    remove_columns=unused_subset.column_names
)

# Combine generated and unused subsets into a new dataset with only the 'text' feature
combined_text_dataset = concatenate_datasets([unused_subset, generated_subset]) if generate_percentage > 0 else unused_subset

# Tokenize the new dataset to prepare for training
# def tokenize_combined_texts(examples):
#     tokenized = tokenizer(
#         examples['text'],
#         truncation=True,
#         max_length=64,
#         return_special_tokens_mask=False,
#     )
#     tokenized['labels'] = tokenized['input_ids'].copy()
#     return tokenized

# processed_dataset = combined_text_dataset.map(
#     tokenize_combined_texts,
#     batched=True,
#     remove_columns=['text'],
# )

# Combine with validation and test datasets to create a DatasetDict
dataset_dict = DatasetDict({
    'train': combined_text_dataset, 
    'validation': raw_dataset['validation'], 
    'test': raw_dataset['test']
})

# Step 6: Save the Combined Dataset
dataset_dict.save_to_disk('generated_dataset')