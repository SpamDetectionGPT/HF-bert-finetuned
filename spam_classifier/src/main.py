from data_processor import DataProcessor
from trainer import SpamTrainer
from config import ModelConfig, DataConfig, TrainingConfig
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logging.info("Starting spam classification training...")

    # Initialize configurations
    model_config = ModelConfig()
    data_config = DataConfig()
    training_config = TrainingConfig()

    # Initialize data processor
    data_processor = DataProcessor(data_config, model_config)
    train_dataset, val_dataset, test_dataset = data_processor.prepare_datasets()
    logging.info("Data processing completed")

    # Initialize and train model
    trainer = SpamTrainer(model_config, training_config)
    model_trainer = trainer.train(train_dataset, val_dataset)
    logging.info("Model training completed")

    # Evaluate model
    results = trainer.evaluate(model_trainer, test_dataset)
    logging.info(f"Test results: {results}")

if __name__ == "__main__":
    main() 