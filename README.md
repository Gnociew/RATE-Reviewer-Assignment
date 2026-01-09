# RATE: Reviewer Profiling and Annotation-free Training for Expertise Ranking in Peer Review Systems

This repository contains the reproduction scripts and documentation for the RATE experiment.

## Project Structure

```
.
├── README.md               # This documentation
├── main.py                 # Main entry point for reproduction
├── scripts/                # Core logic scripts
│   └── RATE.py             # Similarity calculation logic
├── train/                  # Training scripts
│   └── training.py         # Original training code
├── evaluation_script.py    # Evaluation metrics calculation
├── requirements.txt        # Python dependencies
├── configs/                # Configuration files
│   └── example_config.yaml # Example configuration
├── data/                   # Datasets
│   ├── evaluations_pc.json # Paper-Centric evaluation data
│   ├── evaluations_rc.json # Reviewer-Centric evaluation data
│   └── keywords.json       # Keyword file
└── predictions/            # Generated prediction files
```

## Usage

### Environment Setup

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Prepare Model

We will provide two model checkpoints on Hugging Face:
- **RATE-0.6B**: [Hugging Face Link](https://huggingface.co/) (Coming Soon)
- **RATE-8B**: [Hugging Face Link](https://huggingface.co/) (Coming Soon)

### Reproduction

We provide a `main.py` script that orchestrates the reproduction process. It uses `scripts/RATE.py` for inference and `evaluation_script.py` for metrics calculation.

**Option 1: Run with Config File**

We have prepared a configuration file for one-click execution.

```bash
python main.py --config configs/example_config.yaml --base_model_path /path/to/your/model_checkpoint
```

This will:
1.  Generate `predictions/RATE_pc.json` and `predictions/RATE_rc.json`.
2.  Automatically run evaluation and print the Accuracy and Loss metrics.

### Evaluation

If you want to run evaluation separately on existing prediction files:

```bash
python evaluation_script.py --pred_paths predictions/RATE_pc.json predictions/RATE_rc.json
```
