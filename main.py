import argparse
import os
import torch
import yaml
import json
from scripts import RATE
import evaluation_script

# Constants
DEFAULT_KEYWORDS_FILE = "data/keywords.json"
DEFAULT_BASE_MODEL = "checkpoint/RATE_8B"
DEFAULT_ADAPTER_PATH = None

def _output_base_for_input(data_path: str) -> str:
    name = os.path.basename(data_path).lower()
    if "pc" in name:
        suffix = "pc"
    elif "rc" in name:
        suffix = "rc"
    else:
        suffix = os.path.splitext(os.path.basename(data_path))[0]
    return f"RATE_{suffix}"

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--keywords_file", type=str, default=None)
    parser.add_argument("--data_path", type=str, nargs="+", default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    # Defaults
    config = {
        "base_model_path": DEFAULT_BASE_MODEL,
        "adapter_path": DEFAULT_ADAPTER_PATH,
        "keywords_file": DEFAULT_KEYWORDS_FILE,
        "data_path": None, # Must be provided
        "output_path": None, # Must be provided
        "device": "auto"
    }
    
    # Load from YAML if provided
    if args.config:
        yaml_config = load_config(args.config)
        config.update(yaml_config)
    
    # Override with CLI args if provided
    if args.base_model_path: config["base_model_path"] = args.base_model_path
    if args.adapter_path: config["adapter_path"] = args.adapter_path
    if args.keywords_file: config["keywords_file"] = args.keywords_file
    if args.data_path: config["data_path"] = args.data_path
    if args.output_path: config["output_path"] = args.output_path
    if args.device: config["device"] = args.device
    
    # Validate required args
    if not config["data_path"]:
        raise ValueError("data_path must be provided via config or CLI")
    if not config["output_path"]:
        raise ValueError("output_path must be provided via config or CLI")
        
    device_arg = config["device"]
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    
    print(f"Using device: {device}")
    print(f"Config used: {json.dumps(config, indent=2)}")

    # 1. Run Prediction (using scripts/RATE.py logic)
    kw_map = RATE.load_keywords_map(config["keywords_file"])
    data_paths = config["data_path"]
    output_path = config["output_path"]
    generated_files = []

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    if len(data_paths) == 1:
        if output_path.endswith(".json"):
            final_output_path = output_path
        else:
            _ensure_dir(output_path)
            final_output_path = os.path.join(output_path, f"{_output_base_for_input(data_paths[0])}.json")
        
        RATE.calculate_similarity(
            model_path=config["base_model_path"],
            data_path=data_paths[0],
            output_path=final_output_path,
            device=device,
            kw_map=kw_map,
            adapter_path=config["adapter_path"]
        )
        generated_files.append(final_output_path)
    else:
        _ensure_dir(output_path)
        for dp in data_paths:
            out_file = os.path.join(output_path, f"{_output_base_for_input(dp)}.json")
            RATE.calculate_similarity(
                model_path=config["base_model_path"],
                data_path=dp,
                output_path=out_file,
                device=device,
                kw_map=kw_map,
                adapter_path=config["adapter_path"]
            )
            generated_files.append(out_file)

    # 2. Run Evaluation (using evaluation_script.py logic)
    if not args.skip_eval:
        print("\n--- Running Evaluation ---\n")
        # Reuse evaluation_script logic
        # We can call _eval_one_file directly
        for path in generated_files:
            if not os.path.exists(path):
                print(f"Warning: Prediction file {path} not found.")
                continue
                
            label, res_loss, res_acc = evaluation_script._eval_one_file(
                path, 
                pred_field="pred_score", 
                ref_field="score"
            )
            print(f"{label}:  loss: {evaluation_script._format_loss(res_loss['loss'])}  acc: {evaluation_script._format_loss(res_acc['accuracy'])}")

if __name__ == "__main__":
    main()
