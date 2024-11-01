

# # 1
# #powershell -ExecutionPolicy Bypass -File myvenv\Scripts\Activate.ps1
# #2 
# #Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# #3 
# #myvenv\Scripts\activate


############################
import os
import sys
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Print Python environment details
print(sys.prefix)

# Set up Hugging Face API key directly
HUGGING_FACE_API_KEY = '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'  # Ensure this is correct and valid

# Model ID for BART-Large-MNLI
model_id = "facebook/bart-large-mnli"

# Download the model file (if needed)
# Uncomment the next line if you want to download the specific file
# model_path = hf_hub_download(repo_id=model_id, filename="pytorch_model.bin", token=HUGGING_FACE_API_KEY)

# Load the model and tokenizer directly from the Hugging Face Hub
model = AutoModelForSequenceClassification.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Test the model setup
print("Model and tokenizer loaded successfully.")

# Load the zero-shot classification pipeline with the model and tokenizer
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Test the classifier
result = classifier("The sky is blue.", candidate_labels=["science", "weather", "art"])
print(result)
