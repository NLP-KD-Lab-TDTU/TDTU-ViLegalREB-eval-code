"""TSV submission format contract: read, write, and validate TREC-style runs.

Contract Specifications:
- Tab-delimited format (TSV) with exactly 6 columns:
    query_id <TAB> Q0 <TAB> chunk_id <TAB> rank <TAB> score <TAB> run_name
- No header line.
- IDs and run_name must be non-empty whitespace-free tokens.
- Rank must be sequential 1-based integers (1, 2, ..., top_k) for each query.
- Document IDs within the same query must be unique.
- Scores must be finite floating-point numbers (not NaN, not Inf).
- Atomic writing: output is written to a temporary file in the destination folder
  and renamed only upon complete success to prevent partial/corrupt files.
"""

import math
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union


def is_token(val: Any) -> bool:
    """Return True if val is a non-empty string without whitespace characters."""
    return isinstance(val, str) and bool(val) and not any(c.isspace() for c in val)


def save_submission(
    results: Dict[str, Sequence[Tuple[str, float]]],
    output_path: Union[str, Path],
    run_name: str,
    top_k: Optional[int] = None,
) -> None:
    """
    Atomically export retrieval results to a TREC TSV file.

    Args:
        results: Mapping from query_id to sequence of (chunk_id, score).
        output_path: Target path to save the TSV run file.
        run_name: Run identifier token.
        top_k: Maximum number of ranked documents per query (default: all provided).
    """
    if not is_token(run_name):
        raise ValueError(f"Run name must be a non-empty token without whitespace, got: {run_name!r}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=out.parent,
            prefix=".run-",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            for qid, ranked_pairs in results.items():
                if not is_token(qid):
                    raise ValueError(f"Query ID must be a non-empty token without whitespace: {qid!r}")

                pairs_to_write = ranked_pairs if top_k is None else ranked_pairs[:top_k]
                seen_docs: Set[str] = set()

                for rank, (cid, score) in enumerate(pairs_to_write, start=1):
                    if not is_token(cid):
                        raise ValueError(f"Chunk ID must be a non-empty token without whitespace: {cid!r}")
                    if cid in seen_docs:
                        raise ValueError(f"Duplicate document '{cid}' found for query '{qid}'")
                    seen_docs.add(cid)

                    score_val = float(score)
                    if not math.isfinite(score_val):
                        raise ValueError(f"Non-finite retrieval score ({score_val}) for query '{qid}', doc '{cid}'")

                    f.write(f"{qid}\tQ0\t{cid}\t{rank}\t{score_val:.9g}\t{run_name}\n")

        tmp_path.replace(out)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# Backwards compatibility alias
save_trec_run = save_submission


def load_submission(
    tsv_path: Union[str, Path]
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Load a TREC TSV submission into a dictionary mapping query_id -> [(chunk_id, score), ...].
    Results are ordered by the recorded rank.
    """
    path = Path(tsv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission file not found: {path}")

    # Temporary storage: qid -> list of (rank, chunk_id, score)
    query_rankings: Dict[str, List[Tuple[int, str, float]]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            cols = line.rstrip("\r\n").split("\t")
            if len(cols) != 6:
                raise ValueError(
                    f"Line {line_no} in {path.name}: Expected 6 tab-separated columns, found {len(cols)}"
                )

            qid, _, cid, rank_str, score_str, _ = cols
            try:
                rank = int(rank_str)
            except ValueError:
                raise ValueError(f"Line {line_no} in {path.name}: Invalid rank '{rank_str}'")

            try:
                score = float(score_str)
            except ValueError:
                raise ValueError(f"Line {line_no} in {path.name}: Invalid score '{score_str}'")

            if qid not in query_rankings:
                query_rankings[qid] = []
            query_rankings[qid].append((rank, cid, score))

    # Sort by rank and return (chunk_id, score)
    results: Dict[str, List[Tuple[str, float]]] = {}
    for qid, items in query_rankings.items():
        items.sort(key=lambda x: x[0])
        results[qid] = [(cid, score) for _, cid, score in items]

    return results


def validate_submission(
    tsv_path: Union[str, Path],
    expected_query_ids: Optional[Iterable[str]] = None,
    valid_chunk_ids: Optional[Set[str]] = None,
    top_k: Optional[int] = 100,
) -> Tuple[bool, List[str]]:
    """
    Validate a submission file against the leaderboard contract.

    Args:
        tsv_path: Path to the TSV file.
        expected_query_ids: If provided, file must contain exactly these queries.
        valid_chunk_ids: If provided, all chunk IDs must belong to this set.
        top_k: If set, each query must have exactly this many ranked documents (rank 1..top_k).

    Returns:
        (is_valid, errors): Tuple of boolean status and list of error descriptions.
    """
    path = Path(tsv_path)
    errors: List[str] = []

    if not path.is_file():
        return False, [f"File not found or not a regular file: {path}"]

    if path.stat().st_size == 0:
        return False, ["Submission file is empty"]

    expected_qids_set = set(expected_query_ids) if expected_query_ids is not None else None
    seen_qids: Set[str] = set()
    query_docs: Dict[str, Set[str]] = {}
    query_ranks: Dict[str, List[int]] = {}
    query_scores: Dict[str, List[float]] = {}
    current_run_name: Optional[str] = None

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw_line = line.rstrip("\r\n")
            if not raw_line.strip():
                errors.append(f"Line {line_no}: Empty line encountered")
                continue

            cols = raw_line.split("\t")
            if len(cols) != 6:
                errors.append(
                    f"Line {line_no}: Expected 6 tab-separated columns, found {len(cols)}"
                )
                continue

            qid, iter_token, cid, rank_str, score_str, run_name = cols

            # Check tokens
            if not is_token(qid):
                errors.append(f"Line {line_no}: Invalid query ID '{qid}' (must be non-empty without whitespace)")
            if not is_token(cid):
                errors.append(f"Line {line_no}: Invalid chunk ID '{cid}' (must be non-empty without whitespace)")
            if not is_token(run_name):
                errors.append(f"Line {line_no}: Invalid run name '{run_name}' (must be non-empty without whitespace)")

            if current_run_name is None:
                current_run_name = run_name
            elif current_run_name != run_name and is_token(run_name):
                errors.append(
                    f"Line {line_no}: Inconsistent run name '{run_name}' (expected '{current_run_name}')"
                )

            # Check rank
            try:
                rank = int(rank_str)
                if rank <= 0:
                    errors.append(f"Line {line_no}: Rank must be positive, got {rank}")
            except ValueError:
                errors.append(f"Line {line_no}: Rank '{rank_str}' is not an integer")
                rank = None

            # Check score
            try:
                score = float(score_str)
                if not math.isfinite(score):
                    errors.append(f"Line {line_no}: Non-finite score '{score_str}'")
            except ValueError:
                errors.append(f"Line {line_no}: Score '{score_str}' is not a valid float")

            # Check valid chunk IDs
            if valid_chunk_ids is not None and cid and cid not in valid_chunk_ids:
                errors.append(f"Line {line_no}: Chunk ID '{cid}' is not in the benchmark corpus release")

            # Duplicate doc check per query
            if qid not in query_docs:
                query_docs[qid] = set()
                query_ranks[qid] = []
                query_scores[qid] = []
                seen_qids.add(qid)

            if cid in query_docs[qid]:
                errors.append(f"Line {line_no}: Duplicate chunk ID '{cid}' for query '{qid}'")
            else:
                query_docs[qid].add(cid)

            if rank is not None:
                query_ranks[qid].append(rank)

    # Check expected query set
    if expected_qids_set is not None:
        missing_qids = expected_qids_set - seen_qids
        extra_qids = seen_qids - expected_qids_set
        if missing_qids:
            errors.append(f"Submission is missing {len(missing_qids)} expected queries (e.g. {sorted(list(missing_qids))[:5]})")
        if extra_qids:
            errors.append(f"Submission contains {len(extra_qids)} unexpected queries (e.g. {sorted(list(extra_qids))[:5]})")

    # Check top_k and rank continuity per query
    for qid, ranks in query_ranks.items():
        if top_k is not None:
            if len(ranks) != top_k:
                errors.append(f"Query '{qid}' has {len(ranks)} documents (expected exactly {top_k})")
            expected_ranks = list(range(1, len(ranks) + 1))
            if sorted(ranks) != expected_ranks:
                errors.append(
                    f"Query '{qid}' ranks are not strictly 1..{len(ranks)}: missing or disordered rank sequence"
                )

    return (len(errors) == 0, errors)


def verify_submission(
    tsv_path: Union[str, Path],
    expected_query_ids: Optional[Iterable[str]] = None,
    valid_chunk_ids: Optional[Set[str]] = None,
    top_k: Optional[int] = 100,
) -> None:
    """
    Validate submission and raise ValueError with all failure details if invalid.
    """
    is_valid, errors = validate_submission(
        tsv_path,
        expected_query_ids=expected_query_ids,
        valid_chunk_ids=valid_chunk_ids,
        top_k=top_k,
    )
    if not is_valid:
        msg = f"Submission verification failed ({len(errors)} issues):\n" + "\n".join(f" - {e}" for e in errors[:20])
        if len(errors) > 20:
            msg += f"\n ... and {len(errors) - 20} more errors."
        raise ValueError(msg)


def main() -> None:
    import argparse
    import json
    import sys

    default_queries_path = Path(__file__).resolve().parent / "data" / "queries.json"

    parser = argparse.ArgumentParser(
        description="Verify a TREC TSV submission file against TDTU-ViLegalREB benchmark rules."
    )
    parser.add_argument(
        "submission_file",
        type=Path,
        help="Path to the TSV submission file to verify (e.g. results/my_model_seed42_run.tsv)",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=default_queries_path,
        help=f"Path to queries.json (default: {default_queries_path})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Expected number of documents per query (default: 100)",
    )

    args = parser.parse_args()

    if not args.submission_file.exists():
        sys.exit(f"[ERROR] Submission file does not exist: {args.submission_file}")

    expected_qids = None
    if args.queries.exists():
        try:
            with open(args.queries, "r", encoding="utf-8") as f:
                queries_data = json.load(f)
            expected_qids = [q["id"] for q in queries_data]
            print(f"[Info] Loaded {len(expected_qids)} benchmark queries from {args.queries}")
        except Exception as e:
            sys.exit(f"[ERROR] Failed to load queries file '{args.queries}': {e}")
    else:
        print(f"[Warning] Queries file '{args.queries}' not found; skipping query coverage check.")

    try:
        verify_submission(
            args.submission_file,
            expected_query_ids=expected_qids,
            top_k=args.top_k,
        )
        print(f"[SUCCESS] Submission '{args.submission_file}' is valid and ready for the Leaderboard!")
    except ValueError as e:
        sys.exit(f"[FAIL] {e}")


if __name__ == "__main__":
    main()

