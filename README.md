# TDTU-ViLegalREB: Vietnamese Legal Information Retrieval Benchmark

This repository contains the official retrieval runner for generating leaderboard submission files.


## Repository Structure

```
.
├── README.md
├── requirements.txt
├── run.py                     # RUN THIS FILE TO GET SUBMISSION FILE
├── submission.py              # Validate submisson file
├── data
└── .gitignore
```

## Dataset

Download benchmark [NLP-KD-Lab-TDTU/TDTU-ViLegalREB](https://huggingface.co/datasets/NLP-KD-Lab-TDTU/TDTU-ViLegalREB) to `data/`

## Installation

Install all required benchmark dependencies with pip:

```bash
pip install -r requirements.txt
```

## Running Retrieval

The public runner performs retrieval over the full corpus (or candidate pool) and outputs an atomic, verified TREC-style TSV file. Evaluation metrics (nDCG@10, Recall, etc.) are computed independently on the official Leaderboard to preserve benchmark integrity.

### 1. Run BM25 Baseline
```bash
python run.py --model bm25 --top-k 100
```

### 2. Run Dense Retrieval
```bash
python run.py \
    --model dense \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --device cuda \
    --top-k 100 \
    --seed 42
```

### 3. Run 2-Stage Retrieval / Reranking
To run retrieval or reranking on the pre-filtered candidate pool (347K chunks instead of the full 3.03M chunks), pass `--corpus-file corpus_candidates.json`:
```bash
python run.py \
    --model dense \
    --encoder-name sentence-transformers/all-MiniLM-L6-v2 \
    --corpus-file corpus_candidates.json \
    --device cuda \
    --top-k 100
```
> **Tip:** The runner automatically isolates caches (`dense_corpus_candidates_...` vs `dense_corpus_...`) and names the submission file `results/{model}_corpus_candidates_seed42_run.tsv` so it never overwrites full benchmark runs.

### CLI Options
- `--model {bm25,dense}`: Select retrieval model (default: `bm25`). Auto download model from HuggingFace if it is dense.
- `--corpus-file FILE`: Corpus filename in `--benchmark-dir` or path (default: `corpus.json`; set to `corpus_candidates.json` for 2-stage mode).
- `--top-k INT`: Number of ranked documents per query (default: `100`).
- `--device {cpu,cuda}`: Compute device for dense encoding.
- `--hf-token TOKEN`: Hugging Face access token for private models (defaults to `$HF_TOKEN`).
- `--query-instruction [TEXT]`: Task instruction prefix for instruction-tuned models (e.g. e5).
- `--force-reindex`: Force index/embedding recomputation if corpus or cache changed.
- `--run-tag TAG`: Custom tag appended to the output file name.

Results will be exported to `results/{run_name}_run.tsv`.

## Leaderboard Submission

All submissions to the Leaderboard must adhere to the 6-column TREC TSV contract:

```tsv
<query_id>	Q0	<chunk_id>	<rank>	<score>	<run_name>
```

### Requirements:
1. **6 Tab-Separated Columns**: Exactly 6 columns delimited by `\t`, no header row.
2. **Tokens**: `query_id`, `chunk_id`, and `run_name` must be non-empty tokens without whitespace.
3. **Sequential Rank**: For every query, ranks must be strictly consecutive integers starting from 1 to `top-k`.
4. **Unique Documents**: No duplicate document IDs within any single query's ranked list.
5. **Finite Scores**: Scores must be valid, finite floating-point numbers (no NaN or Inf).
6. **Query Coverage**: The submission must contain retrieval results for all 143 queries in `queries.json`.

## Validating Your Submission File

Before submitting your TSV file to the leaderboard, verify its format and contract compliance directly using the `submission.py` CLI:

```bash
python submission.py results/my_model_seed42_run.tsv
```

The script automatically:
- Checks column counts (6 columns).
- Checks rank sequences.
- Ensures finite scores and document ID uniqueness.
- Validates complete query coverage against `data/queries.json`.

You can also customize the query file or top-k cutoff:
```bash
python submission.py results/my_model_seed42_run.tsv \
  --queries data/queries.json \
  --top-k 100
```

## Paper

> TBA

## Citation

> TBA

## Contact

- Anh-Cuong Le (Paper): leanhcuong@tdtu.edu.vn
- Quang Huy Nguyen (System): nguyenquanghuy.st@tdtu.edu.vn
