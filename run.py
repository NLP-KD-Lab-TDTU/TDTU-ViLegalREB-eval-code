"""Public retrieval runner: corpus + queries -> TREC TSV submission without qrels."""
import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple

# Thread pool configuration before importing numerical backends
_CPU_THREADS = max(1, int(os.environ.get("TDTU-ViLegalREB_CPU_THREADS", "8")))
os.environ.setdefault("OMP_NUM_THREADS", str(_CPU_THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
from submission import save_submission

_SRC_DIR = Path(__file__).resolve().parent
_SLOW_TOKENIZER_MODELS = {
    "GreenNode/GreenNode-Embedding-E5-Large-VN-V1",
    "intfloat/multilingual-e5-large-instruct",
}


def check_package(package_name: str) -> None:
    """Verifies mandatory library is installed; exits with error otherwise."""
    if importlib.util.find_spec(package_name) is None:
        sys.exit(
            f"[FATAL ERROR] Mandatory package '{package_name}' is not installed.\n"
            f"Install all dependencies via: pip install -r requirements.txt"
        )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Corpus & Query Data Ingestion & Caching
# ---------------------------------------------------------------------------

def corpus_fingerprint(chunks: Sequence[Tuple[str, str]]) -> str:
    """Computes a SHA-256 fingerprint for ordered chunks and validates uniqueness."""
    if not chunks:
        raise ValueError("Corpus cannot be empty")
    seen_ids = set()
    digest = hashlib.sha256()
    for cid, text in chunks:
        if cid in seen_ids:
            raise ValueError(f"Duplicate chunk ID in corpus: {cid}")
        seen_ids.add(cid)
        for val in (cid, text):
            raw = val.encode("utf-8")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
    return digest.hexdigest()


def cache_needs_build(artifact: Path, manifest: Path, signature: Dict[str, Any], force: bool) -> bool:
    """Checks cache existence and signature validity. Fails fast if cache is inconsistent."""
    if force:
        manifest.unlink(missing_ok=True)
        return True
    if not artifact.exists() and not manifest.exists():
        return True
    if not artifact.exists() or not manifest.exists():
        raise ValueError(f"Incomplete cache at {artifact.parent}; run with --force-reindex")
    try:
        cached_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Corrupted cache manifest at {manifest}: {e}; run with --force-reindex")
    if cached_manifest != signature:
        raise ValueError(f"Cache manifest mismatch at {manifest}; run with --force-reindex")
    return False


def write_signature(path: Path, signature: Dict[str, Any]) -> None:
    """Atomically writes cache manifest."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(signature, sort_keys=True, indent=2), encoding="utf-8")
    tmp.replace(path)


def stream_corpus(corpus_path: Path, limit: Optional[int] = None) -> Generator[Tuple[str, str], None, None]:
    """Streams (chunk_id, canonical_text) from the corpus JSON array."""
    if not corpus_path.exists():
        sys.exit(f"[FATAL ERROR] Corpus file does not exist: {corpus_path}")
    import ijson
    with corpus_path.open("rb") as stream:
        for count, item in enumerate(ijson.items(stream, "item")):
            if limit is not None and count >= limit:
                break
            cid, text = item.get("chunk_id"), item.get("canonical_text")
            if not cid or not isinstance(text, str) or not text.strip():
                raise ValueError(f"Corpus item {count} requires non-empty 'chunk_id' and 'canonical_text'")
            yield cid, text


def load_queries(queries_path: Path) -> List[Dict[str, Any]]:
    """Loads query list from queries.json."""
    if not queries_path.exists():
        sys.exit(f"[FATAL ERROR] Queries file does not exist: {queries_path}")
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"Queries file must contain a non-empty array: {queries_path}")
    return data


# ---------------------------------------------------------------------------
# Sparse Retrieval (BM25s)
# ---------------------------------------------------------------------------

def run_bm25(
    corpus_chunks: List[Tuple[str, str]],
    queries: List[Dict[str, Any]],
    cache_dir: Path,
    top_k: int = 100,
    force_reindex: bool = False,
    corpus_tag: str = "corpus",
) -> Dict[str, List[Tuple[str, float]]]:
    """Runs BM25 retrieval using bm25s over the corpus."""
    check_package("bm25s")
    import gc
    import bm25s

    index_dir = cache_dir / f"bm25s_{corpus_tag}_index"
    chunk_ids_file = index_dir / "chunk_ids.npy"
    chunk_ids = [cid for cid, _ in corpus_chunks]

    signature = {
        "corpus_sha256": corpus_fingerprint(corpus_chunks),
        "tokenizer": "bm25s_default_no_stopwords",
        "bm25s_version": bm25s.__version__,
    }
    manifest_file = index_dir / "evaluation_manifest.json"
    should_reindex = cache_needs_build(index_dir, manifest_file, signature, force_reindex)

    if not should_reindex:
        saved_chunk_ids = np.load(str(chunk_ids_file), allow_pickle=False)
        if saved_chunk_ids.tolist() != chunk_ids:
            raise ValueError("BM25 cache IDs mismatch; run with --force-reindex")
        print(f"[BM25] Loading memory-mapped index from {index_dir}...")
        retriever = bm25s.BM25.load(str(index_dir), mmap=True)
    else:
        print(f"[BM25] Building BM25 index over full {len(corpus_chunks):,} documents...")
        corpus_texts = [text for _, text in corpus_chunks]
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=None, show_progress=True)
        del corpus_texts
        gc.collect()

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens, show_progress=True)
        del corpus_tokens
        gc.collect()

        index_dir.mkdir(parents=True, exist_ok=True)
        retriever.save(str(index_dir))
        np.save(str(chunk_ids_file), np.array(chunk_ids))
        write_signature(manifest_file, signature)
        print(f"[BM25] Index saved to {index_dir}")

    top_k = min(top_k, len(corpus_chunks))
    print(f"[BM25] Retrieving top-{top_k} for {len(queries)} queries...")
    query_texts = [q["query"] for q in queries]
    query_tokens = bm25s.tokenize(query_texts, stopwords=None)
    docs, scores = retriever.retrieve(query_tokens, k=top_k, show_progress=True)

    return {
        q["id"]: [(chunk_ids[docs[i, r]], float(scores[i, r])) for r in range(top_k)]
        for i, q in enumerate(queries)
    }


# ---------------------------------------------------------------------------
# Dense Retrieval (SentenceTransformers + FAISS)
# ---------------------------------------------------------------------------

def format_query_with_instruction(
    query: str,
    instruction: Optional[str] = None,
) -> str:
    """Formats an evaluation query with task instruction if specified."""
    if not instruction:
        return query
    return f"Instruct: {instruction}\nQuery: {query}"


def dense_tokenizer_kwargs(model_name: str) -> Dict[str, Any]:
    """Select the legacy backend before loading weights for models with Rust tokenizer issues."""
    if model_name in _SLOW_TOKENIZER_MODELS:
        check_package("sentencepiece")
        return {"tokenizer_kwargs": {"use_fast": False}}
    return {}


def run_dense(
    corpus_chunks: List[Tuple[str, str]],
    queries: List[Dict[str, Any]],
    model_name: str,
    cache_dir: Path,
    batch_size: int = 256,
    top_k: int = 100,
    force_recompute: bool = False,
    query_instruction: Optional[str] = None,
    device: str = "cpu",
    seed: int = 42,
    hf_token: Optional[str] = None,
    corpus_tag: str = "corpus",
) -> Dict[str, List[Tuple[str, float]]]:
    """Runs Dense retrieval using SentenceTransformers & FAISS with strict isolation and caching."""
    for pkg in ("torch", "sentence_transformers", "faiss"):
        check_package(pkg)

    import faiss  # type: ignore
    import gc  # type: ignore
    import torch  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; will not fall back to CPU")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_kwargs: Dict[str, Any] = {}
    if device == "cuda":
        major, minor = torch.cuda.get_device_capability(0)
        gpu_name = torch.cuda.get_device_name(0)
        if major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            dtype = torch.bfloat16
        elif major >= 7:
            dtype = torch.float16
        else:
            dtype = torch.float32
        model_kwargs = {"torch_dtype": dtype}
        print(f"[Dense] Hardware: {gpu_name} (CC {major}.{minor}) -> Dtype: {dtype}")

    print(f"[Dense] Loading model '{model_name}' on device: {device}")
    tokenizer_kwargs = dense_tokenizer_kwargs(model_name)
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        token=hf_token,
        model_kwargs=model_kwargs,
        **tokenizer_kwargs,
    )

    safe_tag = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", model_name)
    emb_cache_file = cache_dir / f"dense_{corpus_tag}_{safe_tag}.npy"
    manifest_file = emb_cache_file.with_suffix(".manifest.json")
    chunk_ids = [cid for cid, _ in corpus_chunks]

    import sentence_transformers # type: ignore
    signature = {
        "corpus_sha256": corpus_fingerprint(corpus_chunks),
        "model": model_name,
        "max_seq_length": model.max_seq_length,
        "normalize_embeddings": True,
        "sentence_transformers": sentence_transformers.__version__,
    }

    should_recompute = cache_needs_build(emb_cache_file, manifest_file, signature, force_recompute)
    if not should_recompute:
        corpus_embeddings = np.load(str(emb_cache_file), mmap_mode="r", allow_pickle=False)
        expected_dim = (
            model.get_embedding_dimension()
            if hasattr(model, "get_embedding_dimension")
            else model.get_sentence_embedding_dimension()
        )
        if corpus_embeddings.ndim != 2 or corpus_embeddings.shape[0] != len(corpus_chunks) or (
            expected_dim and corpus_embeddings.shape[1] != expected_dim
        ):
            raise ValueError("Cached embeddings shape mismatch; run with --force-reindex")
        print(f"[Dense] Loaded cached corpus embeddings from {emb_cache_file}")
    else:
        print(f"[Dense] Encoding {len(corpus_chunks):,} documents (batch_size={batch_size})...")
        corpus_texts = [text for _, text in corpus_chunks]
        corpus_embeddings = model.encode_document(
            corpus_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        del corpus_texts
        gc.collect()

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_cache_file), corpus_embeddings)
        write_signature(manifest_file, signature)
        print(f"[Dense] Embeddings cached to {emb_cache_file}")

    if query_instruction:
        print(f"[Dense] Applying query instruction: '{query_instruction}'")
        query_texts = [
            format_query_with_instruction(q["query"], instruction=query_instruction)
            for q in queries
        ]
    else:
        print(f"[Dense] Encoding {len(queries)} evaluation queries...")
        query_texts = [q["query"] for q in queries]

    query_embeddings = model.encode_query(
        query_texts,
        **({"prompt": ""} if query_instruction else {}),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    corpus_f32 = np.ascontiguousarray(corpus_embeddings, dtype=np.float32)
    query_f32 = np.ascontiguousarray(query_embeddings, dtype=np.float32)

    dim = corpus_f32.shape[1]
    faiss.omp_set_num_threads(min(_CPU_THREADS, os.cpu_count() or 1))
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_f32)

    top_k = min(top_k, len(corpus_chunks))
    scores, indices = index.search(query_f32, top_k)

    return {
        q["id"]: [(chunk_ids[indices[i, r]], float(scores[i, r])) for r in range(top_k)]
        for i, q in enumerate(queries)
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bm25", "dense"], default="bm25", help="Retrieval model backend")
    parser.add_argument("--benchmark-dir", type=Path, default=_SRC_DIR / "data", help="Directory containing corpus & queries")
    parser.add_argument(
        "--corpus-file",
        default="corpus.json",
        help="Corpus filename in benchmark-dir or path (default: corpus.json; use corpus_candidates.json for 2-stage)",
    )
    parser.add_argument("--output-dir", type=Path, default=_SRC_DIR / "results", help="Directory for exported runs")
    parser.add_argument("--cache-dir", type=Path, default=_SRC_DIR / "run_cache", help="Directory for index/embeddings cache")
    parser.add_argument("--top-k", type=int, default=100, help="Number of retrieved documents per query (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing and re-computation")
    parser.add_argument("--run-tag", default=None, help="Custom tag appended to output filename")
    parser.add_argument("--encoder-name", default="sentence-transformers/all-MiniLM-L6-v2", help="Dense model name/path")
    parser.add_argument("--batch-size", type=int, default=256, help="Encoding batch size")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for dense encoder")
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token for private/gated models (defaults to $HF_TOKEN)",
    )
    parser.add_argument(
        "--query-instruction",
        nargs="?",
        const="Given a legal inquiry, retrieve relevant statutory provisions that answer the query",
        default=None,
        help="Query instruction for instruction-tuned models",
    )
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    if args.top_k < 1 or args.batch_size < 1 or not 0 <= args.seed < 2**32:
        parser.error("top-k and batch-size must be positive; seed must be in [0, 2**32)")
    if args.run_tag and not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.run_tag):
        parser.error("run-tag may contain only letters, digits, underscore, dot and hyphen")

    if args.model == "bm25":
        check_package("bm25s")
    else:
        for pkg in ("torch", "sentence_transformers", "faiss"):
            check_package(pkg)

    set_seed(args.seed)

    queries = load_queries(args.benchmark_dir / "queries.json")
    for q in queries:
        qid, qtext = q.get("id"), q.get("query")
        if not qid or any(c.isspace() for c in qid) or not isinstance(qtext, str) or not qtext.strip():
            raise ValueError("All queries require a whitespace-free token 'id' and non-empty 'query' text")
    if len({q["id"] for q in queries}) != len(queries):
        raise ValueError("Duplicate query IDs detected in queries.json")

    corpus_path = Path(args.corpus_file)
    if not corpus_path.is_file():
        corpus_path = args.benchmark_dir / args.corpus_file
    corpus_tag = corpus_path.stem

    corpus = list(stream_corpus(corpus_path))
    if len(corpus) < args.top_k:
        raise ValueError(f"Corpus has fewer documents ({len(corpus)}) than requested top-k ({args.top_k})")

    if args.model == "bm25":
        results = run_bm25(corpus, queries, args.cache_dir, args.top_k, args.force_reindex, corpus_tag=corpus_tag)
        model_tag = "bm25"
    else:
        results = run_dense(
            corpus, queries, args.encoder_name, args.cache_dir,
            batch_size=args.batch_size, top_k=args.top_k, force_recompute=args.force_reindex,
            query_instruction=args.query_instruction, device=args.device, seed=args.seed,
            hf_token=args.hf_token, corpus_tag=corpus_tag,
        )
        model_tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", args.encoder_name)

    if corpus_tag != "corpus":
        model_tag += f"_{corpus_tag}"
    if args.run_tag:
        model_tag += f"_{args.run_tag}"
    run_name = f"{model_tag}_seed{args.seed}"
    out_file = args.output_dir / f"{run_name}_run.tsv"
    save_submission(results, out_file, run_name, top_k=args.top_k)
    print(f"[Output] Exported submission TSV to: {out_file}")


if __name__ == "__main__":
    main()
