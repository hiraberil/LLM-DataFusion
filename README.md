# LLM Data Fusion Pipeline

Official code for the paper:

> **Single and Multi Truth Data Fusion using Large Language Models**  
> Hira Beril Kucuk, Norman W Paton, Jiaoyan Chen, Zhenyu Wu  
> Department of Computer Science, University of Manchester  
> ADBIS 2025

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This pipeline uses Large Language Models (LLMs) to resolve conflicting values from multiple sources — a problem known as data fusion or truth discovery. It supports both **multi-truth** settings (book authors, movie directors) and **single-truth** settings (flight departure/arrival times and gates).

Prompt strategies evaluated:
- **Domain-Dependent (DD)** vs **Domain-Independent (DI)**
- **0-shot** vs **1-shot**
- With and without constraints **C1** (source constraint) and **C2** (deduplication)

## Key Results (GPT-4o-mini)

| Dataset | Best Config | Recall | Precision | F1 |
|---------|-------------|--------|-----------|-----|
| Book | DD-1shot | 0.8102 | 0.7551 | **0.7817** |
| Movie | DD-1shot | 0.8690 | 0.7713 | **0.8172** |
| Flight | DD-C1-0shot | 0.9544 | 0.8731 | **0.9119** |

DD-based LLM prompts outperform all traditional truth discovery baselines (DART, LTM, MV, SRV) across all datasets. See the paper for full results.



## Repository Structure

```
├── config.py                    # Paths, LLM settings, prompt style
├── data/
│   ├── loader.py                # Load raw claims (book, movie, flight)
│   ├── normalizer.py            # Text normalization
│   └── preprocessor.py          # Preprocessing pipeline
├── evaluation/
│   ├── truth_loader.py          # Load ground truth
│   └── metrics.py               # Recall/Precision/F1 
├── llm/
│   ├── client.py                # OpenAI and Anthropic API calls
│   ├── prompt_builder.py        # All prompt variants (DD/DI × 0/1-shot × C1/C2)
│   ├── parser.py                # Parse LLM responses (book, movie)
│   └── parser_flight.py         # Parse LLM responses (flight)
└── run.py                       # Run all experiments (book, movie, flight)
```

## Datasets

Three benchmark datasets are used:

| Dataset | Type | Task |
|---------|------|------|
| Book | Multi-truth | Predict correct author(s) from conflicting seller records |
| Movie | Multi-truth | Predict correct director(s) from conflicting source records |
| Flight | Single-truth | Predict correct departure/arrival times and gates |

Datasets are publicly available at:
- Book & Flight: https://lunadong.com/fusiondatasets
- Movie: https://heathersherry.github.io/

Set the paths to your local copies in `config.py`.

## Setup

### Prerequisites

- Python 3.9+
- An OpenAI or Anthropic API key

### 1. Install dependencies

```bash
pip install openai anthropic
```

### 2. Configure API keys

Edit `.env` and fill in your API keys:

```
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 3. Set data paths

Edit `config.py` and set the paths to your dataset files:

```python
BOOK_CLAIMS_PATH   = ""  # path to book claims file
BOOK_TRUTH_PATH    = ""  # path to book truth file
MOVIE_CLAIMS_PATH  = ""  # path to movie claims file
MOVIE_TRUTH_PATH   = ""  # path to movie truth file
FLIGHT_CLAIMS_PATH = ""  # path to flight claims file
FLIGHT_TRUTH_PATH  = ""  # path to flight truth file
```

## Running Experiments

```bash
# Book and movie (both DD and DI)
python run.py --dataset book
python run.py --dataset movie
python run.py --dataset both

# Flight
python run.py --dataset flight --claims <path> --truth <path>

# Flight with anonymized flight IDs (blind)
python run.py --dataset flight_blind --claims <path> --truth <path>
```

Use `--mode dd`, `--mode di`, or `--mode both` (default) to select prompt type:

```bash
python run.py --dataset book --mode dd
python run.py --dataset flight --mode di --claims <path> --truth <path>
```

Results are saved to `results/`.

## Prompt Variants

Each experiment runs all combinations of:

| Dimension | Options |
|-----------|---------|
| Prompt type | Domain-Dependent (DD), Domain-Independent (DI) |
| Shot | 0-shot, 1-shot |
| Constraints | baseline, +C1 (source constraint), +C2 (dedup), +C1C2 |

**C1**: *"Only use values that appear in the sources above."*  
**C2**: *"If the same value appears in different formats, count them as one."*

## LLM Configuration

Set the model in `config.py`:

```python
LLM_PROVIDER = "openai"       # "openai" | "anthropic"
LLM_MODEL    = "gpt-4o-mini"  # e.g. "gpt-4o", "claude-sonnet-4-6"
```

Primary experiments in the paper use `gpt-4o-mini`. Results with `gpt-4o` and `claude-sonnet-4-6` are also reported in Table 3.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kucuk2025datafusion,
  title     = {Single and Multi Truth Data Fusion using Large Language Models},
  author    = {Kucuk, Hira Beril and Paton, Norman W and Chen, Jiaoyan and Wu, Zhenyu},
  booktitle = {Proceedings of the 29th European Conference on Advances in Databases and Information Systems (ADBIS)},
  year      = {2025},
  publisher = {Springer}
}
```
