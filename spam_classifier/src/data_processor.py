import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from typing import Tuple, Dict
from .config import DataConfig, ModelConfig

class DataProcessor:
    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)

    def load_data(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load and preprocess the dataset."""
        # Load CSV files
        train_df = pd.read_csv(self.data_config.train_data_path)
        test_df = pd.read_csv(self.data_config.test_data_path)

        # Convert to HuggingFace datasets
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)

        # Split train into train and validation
        train_val_split = train_dataset.train_test_split(
            test_size=self.data_config.val_split,
            seed=self.data_config.random_seed
        )
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]

        return train_dataset, val_dataset, test_dataset

    def preprocess_function(self, examples: Dict) -> Dict:
        """Tokenize the text data."""
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.model_config.max_length,
            return_tensors="pt"
        )

    def prepare_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Prepare the datasets for training."""
        train_dataset, val_dataset, test_dataset = self.load_data()
        
        # Tokenize the datasets
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names
        )

        return train_dataset, val_dataset, test_dataset 