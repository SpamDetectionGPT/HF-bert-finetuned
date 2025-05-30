import torch
from transformers import BertForSequenceClassification
import os

# Define the directory containing the model files
model_dir = "." 

# Define the output path for the .pt file
output_pt_path = "model.pt"

try:
    # Load the model from the directory
    # This will automatically look for config.json, model.safetensors (or pytorch_model.bin)
    print(f"Loading model from {os.path.abspath(model_dir)}...")
    model = BertForSequenceClassification.from_pretrained(model_dir)
    print("Model loaded successfully.")

    # Save the model's state_dict
    print(f"Saving model state_dict to {output_pt_path}...")
    torch.save(model.state_dict(), output_pt_path)
    print(f"Model state_dict saved successfully to {os.path.abspath(output_pt_path)}")

    # You can also save the full model if needed, though state_dict is generally preferred
    # output_full_model_pt_path = "full_model.pt"
    # torch.save(model, output_full_model_pt_path)
    # print(f"Full model saved successfully to {os.path.abspath(output_full_model_pt_path)}")

except Exception as e:
    print(f"An error occurred: {e}") 