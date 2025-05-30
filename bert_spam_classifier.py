import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import json

# Check for CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# Load the dataset
df = pd.read_csv("processed_data 2.csv")

# Display the first few rows and info to understand the data
print(df.head())
print(df.info())
print("Label distribution:\n", df['label'].value_counts())

# --- Data Preparation ---

# Select relevant columns
df = df[['subject', 'label']].copy()

# Handle potential missing values in 'subject'
df['subject'] = df['subject'].fillna('') # Replace NaN with empty string

# Map labels to integers (assuming 0 for non-spam, 1 for spam)
# Adjust this mapping based on the actual label values printed above
# Example: If labels are 'ham' and 'spam'
# label_map = {'ham': 0, 'spam': 1}
# If labels are already 0 and 1, you might skip or adjust this
unique_labels = df['label'].unique()
if len(unique_labels) == 2:
    # Simple binary case: map one to 0, the other to 1
    # Ensure consistent mapping, e.g., alphabetically
    sorted_labels = sorted(list(unique_labels))
    label_map = {label: i for i, label in enumerate(sorted_labels)}
    print(f"Mapping labels: {label_map}")
    df['label'] = df['label'].map(label_map)
else:
    print(f"Warning: Expected 2 unique labels, found {len(unique_labels)}. Check label mapping.")
    # Handle other cases or raise an error if needed


# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['subject'].tolist(),
    df['label'].tolist(),
    test_size=0.2, # 20% for validation
    random_state=42, # for reproducibility
    stratify=df['label'].tolist() # Ensure label distribution is similar in both sets
)

# Load BERT tokenizer
print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer loaded.")

# Tokenize the texts
print("Tokenizing texts...")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
print("Tokenization complete.")

# --- Create PyTorch Datasets ---

class SpamDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Ensure all items are tensors
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SpamDataset(train_encodings, train_labels)
val_dataset = SpamDataset(val_encodings, val_labels)

print("PyTorch datasets created.")

# --- Model Loading and Training Setup ---
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load pre-trained BERT model for sequence classification
print("Loading BERT model...")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))
print("Model loaded.")

# Define compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary') # Use 'binary' for binary classification
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for checkpoints and predictions
    num_train_epochs=1,              # total number of training epochs (adjust as needed, start with 1-3)
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    eval_strategy="epoch",         # Try older argument name
    save_strategy="epoch",         # Use consistent naming (might be correct already)
    load_best_model_at_end=True,     # Load the best model found during training at the end
    metric_for_best_model="f1",      # Use F1 score to determine the best model
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# --- Train the Model ---
print("Starting training...")
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