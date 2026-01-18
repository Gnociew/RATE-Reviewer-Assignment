# RATE: Reviewer Profiling and Annotation-free Training for Expertise Ranking in Peer Review Systems

This repository contains the reproduction code for RATE, including inference scripts to generate expertise ranking predictions and an evaluation script to compute metrics.

## Project Structure

```
.
├── README.md               # This documentation
├── download_model.py       # Script to download model checkpoints
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

We recommend using a clean Conda environment. Install dependencies with `requirements.txt`, download a RATE checkpoint from Hugging Face to a local directory, and then run `main.py` for one-click reproduction.

### Create Conda Environment

```
conda create -n RATE python=3.12
conda activate RATE
```

### Install Dependencies

To install dependencies, run:

```
pip install -r requirements.txt
```

### Prepare Model

We provide two model checkpoints on Hugging Face:
- **RATE-0.6B**: [Hugging Face Link](https://huggingface.co/Gnociew/RATE-0.6B) 
- **RATE-8B**: [Hugging Face Link](https://huggingface.co/Gnociew/RATE-8B)


We provide a helper script to download the model checkpoints from Hugging Face. **We highly recommend using the RATE-8B model** as it achieves significantly better performance.

To download the recommended 8B model:

```bash
python download_model.py --model_size 8B
```

To download the 0.6B model:

```bash
python download_model.py --model_size 0.6B
```

The models will be saved in the `checkpoint/` directory by default.

### Reproduction

We provide a `main.py` script that orchestrates the reproduction process. It uses `scripts/RATE.py` for inference and `evaluation_script.py` for metrics calculation.

**Option 1: Run with Config File**

We have prepared a configuration file for one-click execution.

```bash
python main.py --config configs/example_config.yaml --base_model_path checkpoint/RATE_8B
```

This will:
1.  Generate `predictions/RATE_pc.json` and `predictions/RATE_rc.json`.
2.  Automatically run evaluation and print the Accuracy and Loss metrics.

### Evaluation

If you want to run evaluation separately on existing prediction files:

```bash
python evaluation_script.py --pred_paths predictions/RATE_pc.json predictions/RATE_rc.json
```
