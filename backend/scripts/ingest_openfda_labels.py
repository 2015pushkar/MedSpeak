from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.pipeline import MedicalPipeline
from app.tools.openfda import OpenFDATool


async def run() -> dict:
    settings = get_settings()
    openfda_tool = OpenFDATool()
    pipeline = MedicalPipeline(settings, openfda_tool)
    report = await pipeline.medication_rag.ingest_seed_medications(openfda_tool)
    report["persist_directory"] = settings.chroma_persist_directory
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run()), indent=2))
