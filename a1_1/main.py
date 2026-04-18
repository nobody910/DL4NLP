import os
from transformers import TrainingArguments
from A1 import build_tokenizer, A1RNNModelConfig, A1RNNModel, A1Trainer

def read_texts(filename):
    """Utility to read the dataset into a list of strings."""
    with open(filename, 'r', encoding='utf-8') as f:
        # Each non-empty line is treated as a paragraph/sequence
        return [line.strip() for line in f if line.strip()]

def main():
    # File paths (adjust if your text files are in a different folder)
    train_file = "train.txt"
    val_file = "val.txt"

    print("Loading data...")
    train_dataset = read_texts(train_file)
    eval_dataset = read_texts(val_file)
    print(f"Loaded {len(train_dataset)} training items and {len(eval_dataset)} validation items.")

    print("Building tokenizer...")
    # You can tweak max_voc_size and model_max_length
    tokenizer = build_tokenizer(train_file, max_voc_size=10000, model_max_length=64)
    print(f"Vocabulary size: {len(tokenizer)}")

    print("Setting up model...")
    # Define the size of your RNN network
    config = A1RNNModelConfig(
        vocab_size=len(tokenizer),
        embedding_size=128,  # Size of word embeddings
        hidden_size=256      # Size of the RNN hidden state
    )
    model = A1RNNModel(config)

    print("Setting up training arguments...")
    # We use HuggingFace's TrainingArguments to hold our hyperparameters
    args = TrainingArguments(
        output_dir="./saved_model",
        learning_rate=1e-3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=15,
        eval_strategy="epoch", # Validate at the end of each epoch
        optim="adamw_torch",
        use_cpu=False # Set to True if you don't have a GPU and the code crashes
    )
    # Ensure compatibility with our custom A1Trainer
    args.no_cuda = False 

    print("Initializing trainer...")
    trainer = A1Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    print("Starting training loop...")
    trainer.train()
    print("Training complete! Model saved to ./saved_model")

if __name__ == "__main__":
    main()