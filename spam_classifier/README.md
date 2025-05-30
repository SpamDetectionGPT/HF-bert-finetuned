# Spam Classification Project

This project implements a spam classification system using Hugging Face Transformers. It uses a transformer-based model to classify text messages as spam or not spam.

## Project Structure

```
spam_classifier/
├── data/               # Directory for dataset files
├── models/            # Directory for saved models
├── src/               # Source code
│   ├── config.py      # Configuration settings
│   ├── data_processor.py  # Data loading and preprocessing
│   ├── trainer.py     # Model training and evaluation
│   └── main.py        # Main script
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment using uv:
```bash
uv venv .venv
```

2. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset files in the `data` directory:
   - `train.csv`: Training data
   - `test.csv`: Test data

2. Run the training script:
```bash
python src/main.py
```

## Configuration

The project can be configured by modifying the settings in `src/config.py`:
- Model parameters (model name, batch size, learning rate, etc.)
- Data parameters (file paths, validation split, etc.)
- Training parameters (output directory, logging steps, etc.)

## Model

The default model is BERT-base-uncased, but you can change it by modifying the `model_name` parameter in `ModelConfig`.

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- F1 Score
- Precision
- Recall 