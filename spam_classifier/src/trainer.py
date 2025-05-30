from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from datasets import Dataset
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .config import ModelConfig, TrainingConfig
import torch
from torch.optim import AdamW

class SpamTrainer:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_config.model_name,
            num_labels=model_config.num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def train(self, train_dataset: Dataset, val_dataset: Dataset):
        """Train the model with BERT-specific configurations."""
        # Calculate total training steps
        total_steps = len(train_dataset) * self.model_config.num_epochs // self.model_config.batch_size
        
        training_args = TrainingArguments(
            output_dir=self.training_config.output_dir,
            num_train_epochs=self.model_config.num_epochs,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            learning_rate=self.model_config.learning_rate,
            warmup_steps=self.model_config.warmup_steps,
            weight_decay=self.model_config.weight_decay,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            evaluation_strategy=self.training_config.evaluation_strategy,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=True
        )

        # Initialize optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay
        )

        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.model_config.warmup_steps,
            num_training_steps=total_steps
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            optimizers=(optimizer, scheduler)
        )

        trainer.train()
        return trainer

    def evaluate(self, trainer: Trainer, test_dataset: Dataset) -> Dict:
        """Evaluate the model on test data."""
        return trainer.evaluate(test_dataset) 