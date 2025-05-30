import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
from tqdm import tqdm
from torch.amp import autocast
import torch.backends.cudnn as cudnn
import os.path as path
from multiprocessing import freeze_support

def main():
    # Enable CUDA optimizations
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        # Set memory growth
        torch.cuda.empty_cache()
        # Enable TF32 for better performance on Ampere GPUs
        torch.set_float32_matmul_precision('high')
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")

    # --- Data Loading and Validation ---
    def load_json_dataset(path, label):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'dataset' not in data:
            raise ValueError(f"JSON file {path} missing 'dataset' key.")
        entries = data['dataset']
        if not isinstance(entries, list):
            raise ValueError(f"'dataset' in {path} is not a list.")
        texts = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or 'text' not in entry:
                raise ValueError(f"Entry {i} in {path} missing 'text' key.")
            text = entry['text']
            if not text or not isinstance(text, str):
                raise ValueError(f"Entry {i} in {path} has empty or non-string 'text'.")
            texts.append({'subject': text, 'label': label})
        return texts

    ham_path = 'combined_ham.json'
    spam_path = 'combined_spam.json'

    print("Loading ham (negative) examples...")
    ham_data = load_json_dataset(ham_path, 0)
    print(f"Loaded {len(ham_data)} ham examples.")

    print("Loading spam (positive) examples...")
    spam_data = load_json_dataset(spam_path, 1)
    print(f"Loaded {len(spam_data)} spam examples.")

    # Combine and shuffle
    data = ham_data + spam_data
    if not data:
        raise ValueError("No data loaded from JSON files.")
    df = pd.DataFrame(data)

    # Validation: check for missing or empty subjects
    if df['subject'].isnull().any() or (df['subject'].str.strip() == '').any():
        raise ValueError("Some entries have missing or empty 'subject' fields.")

    print("Combined DataFrame info:")
    print(df.info())
    print("Label distribution:\n", df['label'].value_counts())

    # Split data into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['subject'].tolist(),
        df['label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df['label'].tolist()
    )

    # Load BERT tokenizer
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print("Tokenizer loaded.")

    # Check if tokenized data exists before starting tokenization
    tokenized_data_path = 'tokenized_data.pt'
    if path.exists(tokenized_data_path):
        print("Loading pre-tokenized data...")
        tokenized_data = torch.load(tokenized_data_path)
        train_encodings = tokenized_data['train']
        val_encodings = tokenized_data['val']
        print("Pre-tokenized data loaded successfully.")
    else:
        print("Tokenizing texts...")
        BATCH_SIZE = 1000  # Process 1000 texts at a time

        def tokenize_with_progress(texts, desc):
            all_encodings = []
            for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=desc):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )
                all_encodings.append(batch_encodings)
            
            # Combine all batches
            combined_encodings = {
                key: torch.cat([batch[key] for batch in all_encodings])
                for key in all_encodings[0].keys()
            }
            return combined_encodings

        print("Tokenizing training texts...")
        train_encodings = tokenize_with_progress(train_texts, "Training set")

        print("Tokenizing validation texts...")
        val_encodings = tokenize_with_progress(val_texts, "Validation set")

        print("Saving tokenized data...")
        tokenized_data = {
            'train': train_encodings,
            'val': val_encodings
        }
        torch.save(tokenized_data, tokenized_data_path)
        print("Tokenized data saved successfully.")

    print("Tokenization complete.")

    # --- Create PyTorch Datasets ---
    class SpamDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = SpamDataset(train_encodings, train_labels)
    val_dataset = SpamDataset(val_encodings, val_labels)

    print("PyTorch datasets created.")

    # --- Model Loading and Training Setup ---
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    # Load pre-trained BERT model for sequence classification
    print("Loading BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    print("Model loaded.")

    # Define compute_metrics function for evaluation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    # Define training arguments with optimized settings
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        fp16=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        report_to="none",
    )

    # Initialize Trainer with optimized settings
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Move model to GPU and enable mixed precision
    model = model.to(device)
    model.train()

    # --- Train the Model ---
    print("Starting training...")
    with autocast(device_type='cuda', dtype=torch.float16):
        trainer.train()
    print("Training finished.")

    # --- Evaluate the Model ---
    print("Evaluating the final model on the validation set...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save evaluation results to a file
    with open('evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=4)
    print("Evaluation results saved to evaluation_results.json")

    # --- Save the Model and Tokenizer ---
    model_save_path = "./fine_tuned_bert_spam_classifier"
    print(f"Saving model to {model_save_path}...")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Model and tokenizer saved.")

if __name__ == '__main__':
    freeze_support()
    main() 