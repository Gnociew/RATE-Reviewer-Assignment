import argparse
import json
import os
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Constants
DEFAULT_KEYWORDS_FILE = "data/keywords.json"
DEFAULT_MODEL_PATH = None

DEFAULT_AUTHOR_INSTRUCTION = (
    "For the Author (Query):\n"
    "Represent this author's research profile and expertise based on their historical keywords to find relevant academic papers:"
)
DEFAULT_PAPER_INSTRUCTION = (
    "For the Paper (Document):\n"
    "Represent this academic paper's title and abstract for research recommendation and retrieval:"
)

def _get_id(obj: dict) -> str:
    if not isinstance(obj, dict):
        return "unknown_id"
    return obj.get("paper_id") or obj.get("author_id") or obj.get("id") or "unknown_id"

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _output_base_for_input(data_path: str) -> str:
    name = os.path.basename(data_path).lower()
    if "pc" in name:
        suffix = "pc"
    elif "rc" in name:
        suffix = "rc"
    else:
        suffix = os.path.splitext(os.path.basename(data_path))[0]
    return f"RATE_{suffix}"

def load_keywords_map(keywords_file):
    kw_map = {}
    print(f"Loading keywords from {keywords_file}...")
    if not os.path.exists(keywords_file):
        print(f"Warning: Keywords file {keywords_file} not found. Returning empty map.")
        return kw_map
        
    with open(keywords_file, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if 'paper_title' in item and 'keywords' in item:
                        kw_map[item['paper_title']] = item['keywords']
                    elif 'id' in item and 'keywords' in item:
                        kw_map[item['id']] = item['keywords']
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                try:
                    item = json.loads(line)
                    if 'id' in item and 'keywords' in item:
                        kw_map[item['id']] = item['keywords']
                    elif 'paper_title' in item and 'keywords' in item:
                        kw_map[item['paper_title']] = item['keywords']
                except:
                    continue
    print(f"Loaded {len(kw_map)} keywords.")
    return kw_map

def apply_instruction(text, instruction):
    if not instruction:
        return text
    if not text:
        return instruction
    return f"{instruction}\n{text}"

def get_reviewer_text(reviewer_data, kw_map):
    papers = reviewer_data.get('papers', [])
    all_keywords = []
    
    for p in papers:
        title = p.get('title') or p.get('paper_title')
        pid = p.get('id')
        
        if title and title in kw_map:
            all_keywords.append(kw_map[title])
        elif pid and pid in kw_map:
            all_keywords.append(kw_map[pid])
            
    if not all_keywords and papers:
        pass
            
    text = " ".join(all_keywords)
    return text

def get_paper_text(paper_data):
    title = paper_data.get('paper_title') or paper_data.get('title') or ""
    abstract = paper_data.get('abstract') or ""
    return f"{title} {abstract}".strip()

# --- Model Loading Logic ---

def load_st_model_with_adapter(model_path, adapter_path, device):
    adapter_path = os.path.abspath(adapter_path)
    model = SentenceTransformer(adapter_path, device=device)
    return model

class OursEmbedder:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.cache = {}

    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        # Use convert_to_tensor=True for util.cos_sim
        emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
        self.cache[text] = emb
        return emb

def _process_evaluations_pc(data, embedder, kw_map):
    results = []
    
    for ex in tqdm(data, desc="Processing PC"):
        anchor_text = get_paper_text(ex.get("anchor", {}))
        anchor_text = apply_instruction(anchor_text, DEFAULT_PAPER_INSTRUCTION)
        anchor_emb = embedder.encode(anchor_text)

        # Prepare base item structure
        anchor = ex.get("anchor", {})
        positive = ex.get("positive", {})
        negative = ex.get("negative", {})
        
        item = {
            "anchor_id": _get_id(anchor),
            "positive_id": _get_id(positive),
            "negative_id": _get_id(negative),
            "type": ex.get("type", "paper_centric"),
        }
        
        # Calculate scores
        # Positive (Reviewer)
        pos_text = get_reviewer_text(positive, kw_map)
        pos_text = apply_instruction(pos_text, DEFAULT_AUTHOR_INSTRUCTION)
        pos_emb = embedder.encode(pos_text)
        score_pos = util.cos_sim(anchor_emb, pos_emb).item()

        # Negative (Reviewer)
        neg_text = get_reviewer_text(negative, kw_map)
        neg_text = apply_instruction(neg_text, DEFAULT_AUTHOR_INSTRUCTION)
        neg_emb = embedder.encode(neg_text)
        score_neg = util.cos_sim(anchor_emb, neg_emb).item()

        item["positive_score"] = score_pos
        item["negative_score"] = score_neg
        item["positive_ref_score"] = positive.get("score")
        item["negative_ref_score"] = negative.get("score")
        results.append(item)
            
    return results

def _process_evaluations_rc(data, embedder, kw_map):
    results = []
    
    for ex in tqdm(data, desc="Processing RC"):
        anchor_text = get_reviewer_text(ex.get("anchor", {}), kw_map)
        anchor_text = apply_instruction(anchor_text, DEFAULT_AUTHOR_INSTRUCTION)
        anchor_emb = embedder.encode(anchor_text)

        anchor = ex.get("anchor", {})
        positive = ex.get("positive", {})
        negative = ex.get("negative", {})

        item = {
            "anchor_id": _get_id(anchor),
            "positive_id": _get_id(positive),
            "negative_id": _get_id(negative),
            "type": ex.get("type", "reviewer_centric"),
        }

        # Positive (Paper)
        pos_text = get_paper_text(positive)
        pos_text = apply_instruction(pos_text, DEFAULT_PAPER_INSTRUCTION)
        pos_emb = embedder.encode(pos_text)
        score_pos = util.cos_sim(anchor_emb, pos_emb).item()

        # Negative (Paper)
        neg_text = get_paper_text(negative)
        neg_text = apply_instruction(neg_text, DEFAULT_PAPER_INSTRUCTION)
        neg_emb = embedder.encode(neg_text)
        score_neg = util.cos_sim(anchor_emb, neg_emb).item()

        item["positive_score"] = score_pos
        item["negative_score"] = score_neg
        item["positive_ref_score"] = positive.get("score")
        item["negative_ref_score"] = negative.get("score")
        results.append(item)

    return results

def calculate_similarity(
    model_path: str,
    data_path: str,
    output_path: str,
    device: str,
    kw_map: dict,
    adapter_path: str = None
):
    print(f"Loading data from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("data_path must be a JSON list")

    dataset_type = None
    for x in data[:5]:
        if isinstance(x, dict) and "type" in x:
            dataset_type = x.get("type")
            break
    
    if not dataset_type:
        if data and "anchor" in data[0]:
            anchor = data[0]["anchor"]
            if "papers" in anchor:
                dataset_type = "reviewer_centric"
            else:
                dataset_type = "paper_centric"

    print(f"Detected dataset type: {dataset_type}")

    print(f"Loading model from {model_path}...")
    
    if adapter_path:
        embedder_model = load_st_model_with_adapter(model_path, adapter_path, device)
    else:
        # Resolve absolute path to ensure local loading
        abs_model_path = os.path.abspath(model_path)
        print(f"Loading local model from {abs_model_path}...")
        embedder_model = SentenceTransformer(abs_model_path, device=device, trust_remote_code=True)
        
    embedder = OursEmbedder(embedder_model, device)

    base_output_dir = os.path.dirname(output_path) or "."
    _ensure_dir(base_output_dir)
    
    if dataset_type == "paper_centric":
        results = _process_evaluations_pc(data, embedder, kw_map)
    elif dataset_type == "reviewer_centric":
        results = _process_evaluations_rc(data, embedder, kw_map)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type!r}. Expect 'paper_centric' or 'reviewer_centric'.")

    print(f"Saving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to PEFT adapter (optional)")
    parser.add_argument("--keywords_file", type=str, default=DEFAULT_KEYWORDS_FILE)
    parser.add_argument("--data_path", type=str, nargs="+", required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cpu', 'cuda', 'cuda:0'). Default is 'auto'.")
    return parser.parse_args()

def _load_config(config_path: str) -> dict:
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    args = _parse_args()
    config = _load_config(args.config_path)

    model_path = args.model_path or config.get("model_path")
    if not model_path:
        raise ValueError("Please provide --model_path or set it in config.")
    
    adapter_path = args.adapter_path or config.get("adapter_path")
        
    keywords_file = args.keywords_file or config.get("keywords_file")
    
    data_paths = args.data_path
    output_path = args.output_path
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load keywords map once
    kw_map = load_keywords_map(keywords_file)

    if len(data_paths) == 1:
        if output_path.endswith(".json"):
            final_output_path = output_path
        else:
            _ensure_dir(output_path)
            final_output_path = os.path.join(output_path, f"{_output_base_for_input(data_paths[0])}.json")
            
        calculate_similarity(model_path, data_paths[0], final_output_path, device, kw_map, adapter_path)
    else:
        # Multiple inputs -> output_path MUST be a directory
        _ensure_dir(output_path)
        for dp in data_paths:
            out_file = os.path.join(output_path, f"{_output_base_for_input(dp)}.json")
            calculate_similarity(model_path, dp, out_file, device, kw_map, adapter_path)
