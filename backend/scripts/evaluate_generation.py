from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.data.generation_eval import GenerationEvalCase, GENERATION_EVAL_CASES, select_generation_eval_cases
from app.evals.generation_judge import GenerationEvaluationRecord, GenerationJudge, summarize_generation_evaluations
from app.pipeline import MedicalPipeline
from app.tools.openfda import OpenFDATool
from app.utils.extractor import extract_text_from_pdf_bytes, validate_raw_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MedSpeak generation quality with an LLM-as-judge.")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        help="Bundled sample case name to evaluate. Repeat to select multiple. Defaults to all bundled cases.",
    )
    parser.add_argument("--text-file", type=Path, help="Path to a UTF-8 text file to evaluate instead of bundled cases.")
    parser.add_argument("--pdf-file", type=Path, help="Path to a text-based PDF file to evaluate instead of bundled cases.")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="OpenAI model to use as the evaluation judge.")
    parser.add_argument("--include-analysis", action="store_true", help="Include full MedSpeak analysis output in the final JSON.")
    parser.add_argument("--output", type=Path, help="Optional path to write the JSON results.")
    return parser.parse_args()


async def run(arguments: argparse.Namespace) -> dict:
    settings = get_settings()
    openfda_tool = OpenFDATool()
    pipeline = MedicalPipeline(settings, openfda_tool)
    judge = GenerationJudge.from_settings(settings, model=arguments.judge_model)

    cases = _load_cases(arguments, settings)
    records: list[GenerationEvaluationRecord] = []

    for case in cases:
        analysis = await pipeline.analyze(
            text=case.report_text,
            rate_limit_remaining=999,
            daily_limit=999,
            reset_at=datetime.now(UTC).isoformat(),
        )
        judge_result = await judge.evaluate(
            report_text=case.report_text,
            analysis=analysis,
            case_name=case.name,
        )
        records.append(
            GenerationEvaluationRecord(
                case_name=case.name,
                description=case.description,
                analysis=analysis,
                judge_result=judge_result,
                raw_synthesizer_response=pipeline.last_debug.get("synthesizer_response"),
                raw_judge_response=judge.last_raw_response,
            )
        )

    output = {
        "judge_model": arguments.judge_model,
        "sample_cases": [case.name for case in cases],
        "metrics": summarize_generation_evaluations(records),
        "cases": [
            {
                "case_name": record.case_name,
                "description": record.description,
                "analysis_snapshot": _build_analysis_snapshot(record),
                "raw_synthesizer_response": record.raw_synthesizer_response,
                "raw_judge_response": record.raw_judge_response,
                "judge_result": record.judge_result.model_dump(),
                **({"analysis_output": record.analysis.model_dump()} if arguments.include_analysis else {}),
            }
            for record in records
        ],
    }
    return output


def _load_cases(arguments: argparse.Namespace, settings) -> tuple[GenerationEvalCase, ...]:
    if arguments.text_file and arguments.pdf_file:
        raise ValueError("Choose either --text-file or --pdf-file, not both.")
    if arguments.text_file:
        report = validate_raw_text(arguments.text_file.read_text(encoding="utf-8"), settings.request_char_limit)
        return (
            GenerationEvalCase(
                name=arguments.text_file.stem,
                report_text=report.text,
                description=f"Custom text input from {arguments.text_file.name}.",
            ),
        )
    if arguments.pdf_file:
        extracted = extract_text_from_pdf_bytes(arguments.pdf_file.read_bytes(), settings.request_char_limit)
        return (
            GenerationEvalCase(
                name=arguments.pdf_file.stem,
                report_text=extracted.text,
                description=f"Custom PDF input from {arguments.pdf_file.name}.",
            ),
        )
    if arguments.cases:
        return select_generation_eval_cases(arguments.cases)
    return GENERATION_EVAL_CASES


def _build_analysis_snapshot(record: GenerationEvaluationRecord) -> dict:
    analysis = record.analysis
    return {
        "document_type": analysis.document_type,
        "summary": analysis.summary,
        "warnings": analysis.warnings,
        "medications": [
            {
                "name": medication.name,
                "grounding_status": medication.grounding_status,
                "status": medication.status,
            }
            for medication in analysis.medications
        ],
        "sources": analysis.meta.sources,
        "partial_data": analysis.meta.partial_data,
        "processing_trace": analysis.meta.processing_trace.model_dump(),
    }


def _print_raw_sections(result: dict) -> None:
    for case in result.get("cases", []):
        case_name = case.get("case_name", "unknown-case")
        print(f"\n=== CASE: {case_name} ===")
        print("=== RAW SYNTHESIZER RESPONSE ===")
        print(json.dumps(case.get("raw_synthesizer_response"), indent=2))
        print("=== RAW JUDGE RESPONSE ===")
        print(json.dumps(case.get("raw_judge_response"), indent=2))


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(run(args))
    _print_raw_sections(result)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
