from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import os

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load the dataset
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path='../../data/processed/finetuning_data.txt',
    block_size=128
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Training arguments
training_args = TrainingArguments(
    output_dir='./models/llm',
    overwrite_output_dir=True,
    num_train_epochs=1,  
    per_device_train_batch_size=4,
    save_steps=1000,  
    save_total_limit=1,  # Only save the most recent model
    learning_rate=5e-4,  
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Start training
trainer.train()

# Save model's state_dict
model_save_path = '../../models/llm/fine_tuned_llm_model.pth'
if not os.path.exists(os.path.dirname(model_save_path)):
    os.makedirs(os.path.dirname(model_save_path))
torch.save(model.state_dict(), model_save_path)

print(f"Model's state_dict saved to {model_save_path}")
