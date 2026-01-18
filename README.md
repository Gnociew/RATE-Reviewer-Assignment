# RATE: Reviewer Profiling and Annotation-free Training for Expertise Ranking in Peer Review Systems

This repository contains the reproduction code for RATE, including inference scripts to generate expertise ranking predictions and an evaluation script to compute metrics.

## Project Structure

```
├── configs/                 # Configuration files
│   └── example_config.yaml  # Example config for one-click reproduction
├── data/                    # Evaluation datasets and keyword map
│   ├── evaluations_pc.json  # Paper-Centric evaluation data
│   ├── evaluations_rc.json  # Reviewer-Centric evaluation data
│   └── keywords.json        # Paper keywords file
├── predictions/             # Generated prediction files (outputs)
├── scripts/                 # Core logic
│   └── RATE.py              # Similarity calculation logic
├── train/                   # Training code 
│   └── training.py          # Original training code
├── evaluation_script.py     # Evaluation metrics (loss / accuracy)
├── main.py                  # Main entry point for reproduction
├── requirements.txt         # Python dependencies
└── README.md                # This file
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

Download one checkpoint to a local folder, then pass its local path via `--base_model_path`. This repo loads checkpoints from local directories.

Example (expected local layout):

```
checkpoint/
└── RATE_8B/    # put the downloaded model files here
```

Then run with:

```
python main.py --config configs/example_config.yaml --base_model_path checkpoint/RATE_8B
```


### Prepare Data

The default evaluation data files are already included in `./data/`:

- `data/evaluations_pc.json`: paper-centric evaluation pairs
- `data/evaluations_rc.json`: reviewer-centric evaluation pairs
- `data/keywords.json`: keyword mapping used to build reviewer profiling 

If you want to use your own data, make sure each JSON file is a list of examples with `anchor`, `positive`, `negative`, and compatible fields. The script will automatically detect whether the dataset is paper-centric or reviewer-centric.

### Reproduction

We provide a `main.py` script that orchestrates the reproduction process. It uses `scripts/RATE.py` for inference and `evaluation_script.py` for metrics calculation.

**Option 1: One-click run (recommended)**

```
python main.py --config configs/example_config.yaml --base_model_path checkpoint/RATE_8B
```

This will:
1. Generate prediction files under `predictions/` (e.g., `predictions/RATE_pc.json` and `predictions/RATE_rc.json`).
2. Run evaluation and print pairwise loss / accuracy.

**Option 2: Override paths via CLI**

```
python main.py \
  --base_model_path checkpoint/RATE_8B \
  --keywords_file data/keywords.json \
  --data_path data/evaluations_pc.json data/evaluations_rc.json \
  --output_path predictions \
  --device auto
```

If you only want to generate predictions (skip evaluation):

```
python main.py --config configs/example_config.yaml --base_model_path checkpoint/RATE_8B --skip_eval
```

### Evaluation

If you want to run evaluation separately on existing prediction files:

```
python evaluation_script.py --pred_paths predictions/RATE_pc.json predictions/RATE_rc.json
```
