import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from gpt2_dataset import GPT2Dataset  # Import the GPT2Dataset class

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the tokenized dataset
dataset = GPT2Dataset("training_data_tokens.txt")

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="logs",
    logging_steps=100,
)

def data_collator(data):
    input_ids = torch.stack(data)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "labels": labels}

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Train the GPT-2 model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("output/model")
tokenizer.save_pretrained("output/tokenizer")
