# /src/llm/llm_model.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_model(model_path):
    """
    Load the fine-tuned LLM.
    """
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# model = load_model('../models/llm/fine_tuned_llm_model.pth')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def generate_schedule(tokenized_input, model):
    """
    Generate schedule using the fine-tuned LLM.
    """
    # Tokenizing and processing input for the model
    input_ids = tokenizer.encode(tokenized_input, return_tensors='pt')

    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)  # Create a tensor of ones with the same shape as input_ids

    # Generate response using the model
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=60,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id  # Set pad_token_id to the ID of the EOS token
    )


    # Decode and return the model output
    return tokenizer.decode(output[0], skip_special_tokens=True)
