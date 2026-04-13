from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.data.retrieval_eval import RETRIEVAL_EVAL_CASES
from app.pipeline import MedicalPipeline
from app.tools.openfda import OpenFDATool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate medication retrieval hit rate.")
    parser.add_argument(
        "--ingest-missing",
        action="store_true",
        help="Populate the seed medication corpus before evaluation if the collection is empty.",
    )
    return parser.parse_args()


async def run(*, ingest_missing: bool) -> dict:
    settings = get_settings()
    openfda_tool = OpenFDATool()
    pipeline = MedicalPipeline(settings, openfda_tool)
    if ingest_missing and pipeline.medication_rag.store.count() == 0:
        await pipeline.medication_rag.ingest_seed_medications(openfda_tool)

    total = len(RETRIEVAL_EVAL_CASES)
    top_1_hits = 0
    top_3_hits = 0
    miss_buckets = {"alias_mismatch": 0, "corpus_coverage_gap": 0, "weak_chunk_match": 0}
    cases: list[dict] = []

    for case in RETRIEVAL_EVAL_CASES:
        retrieval = await pipeline.medication_rag.retrieve(case.query, top_k=settings.retrieval_top_k)
        canonical_hits = [chunk.canonical_name for chunk in retrieval.chunks]
        top_1_hit = bool(canonical_hits and canonical_hits[0] == case.expected_medication)
        top_3_hit = case.expected_medication in canonical_hits[: settings.retrieval_top_k]
        if top_1_hit:
            top_1_hits += 1
        if top_3_hit:
            top_3_hits += 1
        if not top_3_hit:
            if pipeline.medication_rag.resolve_name(case.query) != case.expected_medication:
                miss_bucket = "alias_mismatch"
            elif not canonical_hits:
                miss_bucket = "corpus_coverage_gap"
            else:
                miss_bucket = "weak_chunk_match"
            miss_buckets[miss_bucket] += 1
        else:
            miss_bucket = None
        cases.append(
            {
                "query": case.query,
                "expected_medication": case.expected_medication,
                "source_document": case.source_document,
                "top_1_hit": top_1_hit,
                "top_3_hit": top_3_hit,
                "retrieved_medications": canonical_hits,
                "miss_bucket": miss_bucket,
            }
        )

    return {
        "top_1_hit_rate": round(top_1_hits / total, 3) if total else 0.0,
        "top_3_hit_rate": round(top_3_hits / total, 3) if total else 0.0,
        "total_eval_cases": total,
        "miss_buckets": miss_buckets,
        "cases": cases,
    }


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(asyncio.run(run(ingest_missing=arguments.ingest_missing)), indent=2))
