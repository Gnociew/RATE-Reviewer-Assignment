# RATE: Reviewer Assignment with Text Embeddings (Reproduction)

This repository contains the reproduction scripts and documentation for the RATE experiment, utilizing a **Qwen3-Embedding-8B** base model with a specialized LoRA adapter.

## 📁 Project Structure

```
.
├── README_RATE.md          # This documentation
├── main.py                 # Main entry point for reproduction
├── scripts/                # Core logic scripts
│   └── RATE.py             # Similarity calculation logic
├── train/                  # Training scripts
│   └── training.py         # Original training code
├── evaluation_script.py    # Evaluation metrics calculation
├── requirements.txt        # Python dependencies
├── configs/                # Configuration files
│   └── example_config.yaml # Example configuration
├── checkpoint/             # Model checkpoints
│   ├── RATE_0.6B           # RATE-0.6B
│   ├── best_QWEN_RATE_8B   # RATE-8B
├── data/                   # Evaluation datasets
│   ├── evaluations_pc.json # Paper-Centric evaluation data
│   ├── evaluations_rc.json # Reviewer-Centric evaluation data
│   └── keywords.json       # Keyword mapping file
└── predictions/            # Generated prediction files
```

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Model & Data Paths

We use the locally RATE_8B.

*   **Model Path**: `checkpoint/RATE_8B`
*   **Data Files**: `data/evaluations_pc.json`, `data/evaluations_rc.json`

### 3. Reproduction Steps

We provide a `main.py` script that orchestrates the reproduction process. It uses `scripts/RATE.py` for inference and `evaluation_script.py` for metrics calculation.


**Option 1: Run with Config File **

We have prepared a configuration file for one-click execution.

```bash
python main.py --config configs/example_config.yaml
```

This will:
1.  Generate `predictions/RATE_pc.json` and `predictions/RATE_rc.json`.
2.  Automatically run evaluation and print the Accuracy and Loss metrics.

### 4. Evaluation

If you want to run evaluation separately on existing prediction files:

```bash
python evaluation_script.py --pred_paths predictions/RATE_pc.json predictions/RATE_rc.json
```
