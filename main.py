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

trainer.train()

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

# Step 5: Generate New Data
generate_percentage = 0.1  # Adjust as needed
num_examples = int(len(train_dataset) * generate_percentage)
subset_dataset = train_dataset.select(range(num_examples))

def generate_batch_texts(examples):
    input_ids = examples['input_ids']
    # Ensure input_ids are tensors of equal length
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
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    generated_ids = outputs[:, input_ids.shape[1]:]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {'generated_text': generated_texts}


print("Start Generating!!")
subset_dataset = subset_dataset.map(generate_batch_texts, batched=True, batch_size=8)

def tokenize_generated_texts(examples):
    tokenized = tokenizer(
        examples['generated_text'],
        truncation=True,
        max_length=64,
        return_special_tokens_mask=False,
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized

new_dataset = subset_dataset.map(
    tokenize_generated_texts,
    batched=True,
    remove_columns=subset_dataset.column_names,
)

# Step 6: Save the Generated Data
new_dataset.save_to_disk('generated_dataset')