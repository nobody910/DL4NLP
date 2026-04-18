import torch, nltk, pickle, math
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
import matplotlib.pyplot as plt

###
### Part 1. Tokenization.
###
def lowercase_tokenizer(text):
    return [t.lower() for t in nltk.word_tokenize(text)]

def build_tokenizer(train_file, tokenize_fun=lowercase_tokenizer, max_voc_size=None, model_max_length=None,
                    pad_token='<PAD>', unk_token='<UNK>', bos_token='<BOS>', eos_token='<EOS>'):
    """ Build a tokenizer from the given file. """

    # 1. Read texts and count token frequencies
    counter = Counter()
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            text = line.strip()
            if text:
                counter.update(tokenize_fun(text))

    # 2. Add special tokens first
    special_tokens = [pad_token, unk_token, bos_token, eos_token]
    
    # 3. Limit vocabulary size
    limit = max_voc_size - len(special_tokens) if max_voc_size is not None else None
    common_words = [word for word, freq in counter.most_common(limit)]
    
    vocab = special_tokens + common_words
    str_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_str = {i: word for i, word in enumerate(vocab)}

    # 4. Return the configured Tokenizer
    return A1Tokenizer(str_to_int, int_to_str, model_max_length, 
                       pad_token, unk_token, bos_token, eos_token)


class A1Tokenizer:
    """A minimal implementation of a tokenizer similar to tokenizers in the HuggingFace library."""

    def __init__(self, str_to_int, int_to_str, model_max_length, pad_token, unk_token, bos_token, eos_token):
        self.str_to_int = str_to_int
        self.int_to_str = int_to_str
        self.model_max_length = model_max_length
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Compulsory attributes
        self.pad_token_id = str_to_int[pad_token]     
        self.unk_token_id = str_to_int[unk_token]
        self.bos_token_id = str_to_int[bos_token]
        self.eos_token_id = str_to_int[eos_token]

    def __call__(self, texts, truncation=False, padding=False, return_tensors=None):
        if return_tensors and return_tensors != 'pt':
            raise ValueError('Should be pt')
        
        encoded_texts = []
        attention_masks = []
        max_length_in_batch = 0

        # Encode strings to integers
        for text in texts:
            tokens = lowercase_tokenizer(text)
            token_ids = [self.bos_token_id] + [self.str_to_int.get(t, self.unk_token_id) for t in tokens] + [self.eos_token_id]
            
            if truncation and self.model_max_length is not None:
                token_ids = token_ids[:self.model_max_length]
            
            encoded_texts.append(token_ids)
            max_length_in_batch = max(max_length_in_batch, len(token_ids))

        # Padding logic
        for token_ids in encoded_texts:
            mask = [1] * len(token_ids)
            if padding:
                pad_len = max_length_in_batch - len(token_ids)
                token_ids.extend([self.pad_token_id] * pad_len)
                mask.extend([0] * pad_len)
            attention_masks.append(mask)

        # Output format
        if return_tensors == 'pt':
            input_ids = torch.tensor(encoded_texts, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        else:
            input_ids = encoded_texts
            attention_mask = attention_masks

        return BatchEncoding({'input_ids': input_ids, 'attention_mask': attention_mask})

    def __len__(self):
        """Return the size of the vocabulary."""
        return len(self.str_to_int)
    
    def save(self, filename):
        """Save the tokenizer to the given file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_file(filename):
        """Load a tokenizer from the given file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
   

###
### Part 3. Defining the model.
###

class A1RNNModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the RNN-based language model."""
    def __init__(self, vocab_size=10000, embedding_size=128, hidden_size=256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size

class A1RNNModel(PreTrainedModel):
    """The neural network model that implements a RNN-based language model."""
    config_class = A1RNNModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.rnn = nn.LSTM(config.embedding_size, config.hidden_size, batch_first=True)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size)

        # Note: -100 is the value HuggingFace conventionally uses to refer to tokens
        # where we do not want to compute the loss.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, labels=None):
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        logits = self.unembedding(rnn_out)
        
        loss = None
        if labels is not None:
            # Shift the logits and labels to predict the NEXT token.
            # Logits exclude the last step; labels exclude the first step.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # CrossEntropy expects (Batch * SequenceLength, VocabSize)
            loss = self.loss_func(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(logits=logits, loss=loss)


###
### Part 4. Training the language model.
###

class A1Trainer:
    """A minimal implementation similar to a Trainer from the HuggingFace library."""

    def __init__(self, model, args, train_dataset, eval_dataset, tokenizer):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        assert(args.optim == 'adamw_torch')
        assert(args.eval_strategy == 'epoch')

    def select_device(self):
        if self.args.use_cpu:
            return torch.device('cpu')
        if not self.args.no_cuda and torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
            
    def train(self):
        """Train the model."""
        args = self.args
        device = self.select_device()
        print('Device:', device)
        self.model.to(device)
        
        # Configure optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)

        train_loader = DataLoader(self.train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        val_loader = DataLoader(self.eval_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False)
        
        # 新增：用于记录每轮 Loss 的列表
        history_train_loss = []
        history_val_loss = []
        history_val_ppl = []

        for epoch in range(int(args.num_train_epochs)):
            print(f"--- Epoch {epoch+1} / {int(args.num_train_epochs)} ---")
            self.model.train()
            total_train_loss = 0
            
            for batch in train_loader:
                encoded = self.tokenizer(batch, truncation=True, padding=True, return_tensors='pt')
                input_ids = encoded['input_ids'].to(device)
                labels = input_ids.clone()
                labels[labels == self.tokenizer.pad_token_id] = -100
                
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            avg_train_loss = total_train_loss / len(train_loader)
            history_train_loss.append(avg_train_loss)

            # EVALUATION LOOP
            if args.eval_strategy == 'epoch':
                self.model.eval()
                total_eval_loss = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        encoded = self.tokenizer(val_batch, truncation=True, padding=True, return_tensors='pt')
                        input_ids = encoded['input_ids'].to(device)
                        labels = input_ids.clone()
                        labels[labels == self.tokenizer.pad_token_id] = -100
                        
                        outputs = self.model(input_ids=input_ids, labels=labels)
                        total_eval_loss += outputs.loss.item()
                
                avg_val_loss = total_eval_loss / len(val_loader)
                history_val_loss.append(avg_val_loss)
                
                # Perplexity = exp(CrossEntropyLoss)
                val_perplexity = math.exp(avg_val_loss)
                history_val_ppl.append(val_perplexity)
                
                print(f"Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f} | Validation Perplexity: {val_perplexity:.4f}")

        print(f'Saving model to {args.output_dir}.')
        self.model.save_pretrained(args.output_dir)

        # Loss 
        print("Generating loss curve plot...")
        epochs = range(1, int(args.num_train_epochs) + 1)
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history_train_loss, 'b-o', label='Training Loss')
        plt.plot(epochs, history_val_loss, 'r-s', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        
        loss_plot_path = os.path.join(args.output_dir, "loss_curve.png") if os.path.exists(args.output_dir) else "loss_curve.png"
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"Loss curve saved to {loss_plot_path}")

        # Perplexity
        print("Generating perplexity curve plot...")
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, history_val_ppl, 'g-^', label='Validation Perplexity')
        plt.title('Validation Perplexity')
        plt.xlabel('Epochs')
        plt.ylabel('Perplexity')
        plt.xticks(epochs)
        plt.legend()
        plt.grid(True)
        
        ppl_plot_path = os.path.join(args.output_dir, "perplexity_curve.png") if os.path.exists(args.output_dir) else "perplexity_curve.png"
        plt.savefig(ppl_plot_path)
        plt.close() 
        print(f"Perplexity curve saved to {ppl_plot_path}")