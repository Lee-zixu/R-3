# $\mathbb{R}^3$: Composed Video Retrieval via Reasoning-Guided Recalling and Re-ranking

Official submission code for the CoVR-R challenge (CVPR 2026).

## Overview

Two-stage pipeline for Composed Video Retrieval:

1. **Embedding Retrieval** — Encode query/gallery with `Qwen3-VL-Embedding-8B`, retrieve top-K via cosine similarity
2. **Cross-Encoder Reranking** — Rerank with `Qwen3-VL-Reranker-8B` (SentenceTransformers CrossEncoder)

Optional: reasoning-augmented retrieval using `Qwen3-VL-32B-Thinking` (automatically handled inside `submit`).

## Project Structure

```
CoVR-R-submission/
├── validate.py                 # main entry: embed-gallery / submit / rerank-submission
├── utils.py                    # Tools
├── config.py                   # Config dataclass
├── data.py                     # VideoDataset & collate
├── model.py                    # EmbeddingExtractor & VideoProcessor
├── evaluate.py                 # CoVR-R original evaluation
├── prompts.py                  # Prompt templates
├── requirements.txt
├── configs/
│   ├── test_submission.yaml    # test set config
│   └── validation.yaml         # validation set config
├── test-set_no-labels.json     # test queries (download from official link first)
└── val-set_no-labels.json      # val queries (download from official link first)
```

## Environment Setup

```bash
conda create -n r3 python=3.11 -y
conda activate r3
pip install -r requirements.txt
```

## Data Preparation

1. Download CoVR-R video data (WebVid / SS2)
2. Edit `video_dir` in `configs/test_submission.yaml` or `configs/validation.yaml`
3. Model weights auto-download from HuggingFace on first run; use local paths by modifying `embedding_model_name` / `reranker_model_name` in config

## Running

Three steps to generate a submission:

```bash
# Step 1: Encode gallery videos (run once, skips existing npz)
python validate.py embed-gallery --config configs/test_submission.yaml

# Step 2: Generate reasoning traces + retrieve (reasoning-augmented embedding retrieval)
python validate.py submit --config configs/test_submission.yaml --no-rerank --reasoning-for-retrieval

# Step 3: Rerank the retrieval results
python validate.py rerank-submission \
    --config configs/test_submission.yaml \
    --input artifacts/test_submission.json \
    --output artifacts/test_submission_reranked.json
```


## Output Format

```json
[
  {
    "webvid": [
      {
        "id": 12345,
        "video_source": "112/1016223889",
        "video_target": ["112/9876543210", ...],
        "reasoning trace": ["..."]
      }
    ]
  },
  {
    "ss2": [
      {
        "id": 67890,
        "video_source": "74225",
        "video_target": ["12345", ...],
        "reasoning trace": ["..."]
      }
    ]
  }
]
```
