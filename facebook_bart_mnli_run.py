# this code will run the models from  the following location and produces the result  

import os
import socket
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def block_internet():
    def deny(*args, **kwargs):
        raise socket.error("Internet access is blocked.")
    socket.socket = deny

# Block internet access
block_internet()

# Local path where the model files are saved
local_model_path = r"C:\Users\Rashed1236\Documents\HF_ML_facebook_bart_large_mnli"

# Print Python environment details
print(f"Python environment: {sys.prefix}")

# Load the model and tokenizer from the local path
model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Initialize the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Test the classifier
result = classifier(
    "The sky is blue with shining sun. it is a great weather to dance in the stadium in front of thousands of people", 
    candidate_labels=["science", "weather", "art"]
)
print(result)
