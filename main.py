import os
import torch
from transformers import TrainingArguments
from A1 import build_tokenizer, A1Trainer
from A2 import A2ModelConfig, A2Transformer, generate_text

def read_texts(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    train_file = "train.txt"
    val_file = "val.txt"

    print("Loading data...")
    train_dataset = read_texts(train_file)
    eval_dataset = read_texts(val_file)
    print(f"Loaded {len(train_dataset)} training items.")

    print("Building tokenizer...")
    tokenizer = build_tokenizer(train_file, max_voc_size=10000, model_max_length=64)
    print(f"Vocabulary size: {len(tokenizer)}")

    print("Setting up A2 Transformer model...")
    config = A2ModelConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=4,
        num_hidden_layers=2, 
        max_position_embeddings=128 
    )
    model = A2Transformer(config)

    print("Setting up training arguments...")
    args = TrainingArguments(
        output_dir="./saved_a2_model",
        learning_rate=5e-4, 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        eval_strategy="epoch",
        optim="adamw_torch",
        use_cpu=False
    )
    args.no_cuda = False

    trainer = A1Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    print("\n--- Testing Text Generation ---")
    
    device = trainer.select_device()
    model.to(device)

    prompts = [
        "In natural language processing, a Transformer",
        "Is Stockholm the capital of Sweden? Answer yes or no. The answer is",
        "Write a Python program that"
    ]

    for prompt in prompts:
        print(f"\n[Prompt]: {prompt}")
        generated = generate_text(
            model=model, 
            tokenizer=tokenizer, 
            prompt=prompt, 
            max_length=20,     
            temperature=0.8,          
            topk=5,                 
            device=device
        )
        print(f"[Generated]: {generated}")

if __name__ == "__main__":
    main()