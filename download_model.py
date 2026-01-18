import argparse
import os
import sys
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_URLS = {
    "0.6B": "Gnociew/RATE-0.6B",
    "8B": "Gnociew/RATE-8B"
}

def download_model(model_size, cache_dir):
    repo_id = MODEL_URLS.get(model_size)
    if not repo_id:
        print(f"Unknown model size: {model_size}. Available options: {list(MODEL_URLS.keys())}")
        sys.exit(1)
    
    print(f"Downloading {repo_id} to {cache_dir}...")
    local_dir = os.path.join(cache_dir, f"RATE_{model_size}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Model downloaded successfully to: {local_dir}")
    return local_dir

def main():
    parser = argparse.ArgumentParser(description="Download RATE model checkpoints.")
    parser.add_argument("--model_size", type=str, choices=["0.6B", "8B"], default="8B", help="Model size to download (default: 8B).")
    parser.add_argument("--cache_dir", type=str, default="checkpoint", help="Directory to save the model (default: checkpoint).")

    args = parser.parse_args()

    # Ensure cache directory exists
    os.makedirs(args.cache_dir, exist_ok=True)

    try:
        download_model(args.model_size, args.cache_dir)
    except Exception as e:
        print(f"An error occurred during download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
