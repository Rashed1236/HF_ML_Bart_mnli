# this code will download the models into the following location 
from huggingface_hub import hf_hub_download

# Define model ID and local directory for storing model files
model_id = "facebook/bart-large-mnli"
local_dir = r"C:\Users\Rashed1236\Documents\HF_ML_facebook_bart_large_mnli"

# Files typically needed for running the model locally
files_to_download = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json"  # May not exist in every model repo
]

# Loop through each file and download it to the specified local directory
for file_name in files_to_download:
    try:
        hf_hub_download(repo_id=model_id, filename=file_name, local_dir=local_dir)
        print(f"Downloaded {file_name}")
    except Exception as e:
        print(f"File {file_name} could not be downloaded. Error: {e}")

print("Download process completed.")
