# Step 1: Install Required Dependencies
!pip install unsloth datasets accelerate huggingface_hub trl torch

# Step 2: Import Libraries
from unsloth import FastLanguageModel
from datasets import load_dataset
from huggingface_hub import notebook_login
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import os

# Step 3: Login to Hugging Face (Run this cell first)
notebook_login()

# Step 4: Load Model with Unsloth
model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"  # Good for sentiment analysis on T4 GPU

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)

# Step 5: Add LoRA Adapters for Efficient Fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Step 6: Load and Format a WORKING Sentiment Dataset
print("Loading IMDB sentiment dataset...")
# Using IMDB dataset which is well-maintained and reliable
dataset = load_dataset("imdb")

def format_sentiment_analysis(example):
    sentiment = "positive" if example["label"] == 1 else "negative"
    
    # Format as tweet-like text for consistency with your original intent
    messages = [
        {"role": "user", "content": f"Analyze the sentiment of this text: {example['text'][:280]}"},  # Limit to tweet length
        {"role": "assistant", "content": f"The sentiment is {sentiment}."}
    ]
    
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": formatted_text}

print("Formatting dataset for Qwen...")
formatted_dataset = dataset.map(format_sentiment_analysis)

# Use a subset for faster training (adjust based on your needs)
train_dataset = formatted_dataset["train"].select(range(5000))  # 5000 samples for training

print(f"Training dataset size: {len(train_dataset)}")

# Step 7: Set Up Training Arguments
training_args = TrainingArguments(
    output_dir="./sentiment-analysis-qwen",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=150,  # Adjust based on your needs
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="none",
    save_strategy="steps",
    save_steps=50,
)

# Step 8: Create Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=512,
    packing=False,
    args=training_args,
)

# Step 9: Start Training
print("Starting fine-tuning...")
trainer.train()

print("Training completed successfully!")

# Step 10: Save the Fine-tuned Model Locally
print("Saving model locally...")
model.save_pretrained("./sentiment-analysis-qwen-lora")
tokenizer.save_pretrained("./sentiment-analysis-qwen-lora")

# Step 11: Merge with Base Model for Better Performance
print("Merging model with base weights...")
model.save_pretrained_merged(
    "sentiment-analysis-qwen-merged",
    tokenizer,
    save_method="merged_16bit"  # Full precision merge
)

# Step 12: Push to Hugging Face Hub
print("Pushing model to Hugging Face...")

# Replace with your desired repository name
repo_name = "Abhishek1205/qwen-4b-sentiment-analysis"  # Using your username

# Push merged model
model.push_to_hub_merged(
    repo_name,
    tokenizer=tokenizer,
    save_method="merged_16bit",
    token="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"  # Uncomment if needed
)

print(f"Model successfully pushed to: https://huggingface.co/{repo_name}")

# Step 13: Test the Fine-tuned Model
def test_sentiment_analysis(model, tokenizer, text):
    """Test the fine-tuned sentiment analysis model"""
    
    prompt = f"Analyze the sentiment of this text: {text}"
    
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract assistant response
    if "assistant" in response:
        return response.split("assistant")[-1].strip()
    else:
        return response

# Test with sample texts (tweet-like format)
print("\nTesting the fine-tuned model:")
test_texts = [
    "I love this product! It's amazing!",
    "This is the worst service I've ever experienced.",
    "The weather today is okay, nothing special.",
    "Just had a great meeting with the team!",
    "Absolutely terrible customer support",
    "Neutral experience overall"
]

for text in test_texts:
    result = test_sentiment_analysis(model, tokenizer, text)
    print(f"Text: {text}")
    print(f"Analysis: {result}")
    print("-" * 60)

print("\nFine-tuning and deployment completed successfully!")
print(f"Your model is available at: https://huggingface.co/{repo_name}")
