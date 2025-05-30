from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 128  # BERT's optimal sequence length for most tasks
    batch_size: int = 32  # Increased batch size for better GPU utilization
    learning_rate: float = 2e-5  # Standard learning rate for BERT fine-tuning
    num_epochs: int = 3  # Standard number of epochs for BERT fine-tuning
    num_labels: int = 2  # spam vs not spam
    warmup_steps: int = 500  # BERT-specific warmup steps
    weight_decay: float = 0.01  # BERT-specific weight decay

@dataclass
class DataConfig:
    train_data_path: str = "../data/train.csv"
    test_data_path: str = "../data/test.csv"
    val_split: float = 0.1
    random_seed: int = 42

@dataclass
class TrainingConfig:
    output_dir: str = "../models"
    logging_steps: int = 100
    save_steps: int = 500
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"  # Using F1 score as the main metric for spam detection 