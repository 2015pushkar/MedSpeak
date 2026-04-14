from __future__ import annotations

from app.evals.generation_judge import (
    GenerationEvaluationRecord,
    GenerationJudgeResult,
    JudgeBinaryCheck,
    MedicationJudgeCheck,
    build_generation_judge_payload,
    summarize_generation_evaluations,
)
from app.schemas import (
    AnalysisMeta,
    AnalysisResponse,
    DiagnosisResult,
    MedicationEvidence,
    MedicationResult,
    ProcessingTrace,
)
from tests.sample_reports import MIXED_REPORT_TEXT


def build_analysis_response() -> AnalysisResponse:
    return AnalysisResponse(
        document_type="mixed",
        summary="The report includes elevated glucose and a blood pressure medication.",
        warnings=["Some lab values appear outside the listed reference range."],
        labs=[],
        medications=[
            MedicationResult(
                name="Lisinopril",
                purpose="Lisinopril is used to treat hypertension.",
                common_side_effects=["Dizziness."],
                cautions=["Monitor blood pressure and kidney function during therapy."],
                fda_enriched=True,
                grounding_status="rag",
                status="current",
                grounding_note="grounded from local corpus",
                evidence=[
                    MedicationEvidence(
                        source="json",
                        label_section="indications_and_usage",
                        chunk_id="lisinopril-indications-0",
                        snippet="Lisinopril is indicated for the treatment of hypertension.",
                    )
                ],
            )
        ],
        diagnoses=[DiagnosisResult(term="Hypertension", plain_language="High blood pressure.")],
        vitals=[],
        allergies=[],
        surgeries=[],
        risk_factors=[],
        questions_for_doctor=["What might explain my elevated glucose result?"],
        disclaimer="Educational use only.",
        meta=AnalysisMeta(
            rate_limit_remaining=4,
            daily_limit=5,
            rate_limit_reset_at="2099-01-01T00:00:00+00:00",
            partial_data=False,
            partial_data_reasons=[],
            fallback_used=False,
            sources=["json"],
            processing_trace=ProcessingTrace(
                classifier="llm",
                medications="llm",
                diagnoses="llm",
                synthesis="llm",
            ),
        ),
    )


def test_build_generation_judge_payload_includes_grounding_and_trace():
    analysis = build_analysis_response()

    payload = build_generation_judge_payload(
        report_text=MIXED_REPORT_TEXT,
        analysis=analysis,
        case_name="mixed-discharge",
    )

    assert payload["case_name"] == "mixed-discharge"
    assert payload["original_report"] == MIXED_REPORT_TEXT
    assert payload["medspeak_output"]["medications"][0]["grounding_status"] == "rag"
    assert payload["medspeak_output"]["medications"][0]["evidence"][0]["label_section"] == "indications_and_usage"
    assert payload["medspeak_output"]["meta"]["processing_trace"]["medications"] == "llm"


def test_summarize_generation_evaluations_aggregates_binary_metrics():
    analysis = build_analysis_response()
    passing = GenerationJudgeResult(
        summary_faithfulness=JudgeBinaryCheck(passed=True),
        medication_checks=[MedicationJudgeCheck(medication="Lisinopril", grounding_status="rag", supported=True, safe=True)],
        safety_check=JudgeBinaryCheck(passed=True),
        question_quality=JudgeBinaryCheck(passed=True),
        unsupported_claims=[],
        overall_passed=True,
    )
    failing = GenerationJudgeResult(
        summary_faithfulness=JudgeBinaryCheck(passed=False, issues=["Summary overstated the report."]),
        medication_checks=[MedicationJudgeCheck(medication="Lisinopril", grounding_status="rag", supported=False, safe=True)],
        safety_check=JudgeBinaryCheck(passed=True),
        question_quality=JudgeBinaryCheck(passed=False, issues=["Questions missed the main issue."]),
        unsupported_claims=["The summary claimed a diagnosis not present in the report."],
        overall_passed=False,
    )

    summary = summarize_generation_evaluations(
        [
            GenerationEvaluationRecord(case_name="pass-case", analysis=analysis, judge_result=passing),
            GenerationEvaluationRecord(case_name="fail-case", analysis=analysis, judge_result=failing),
        ]
    )

    assert summary["total_cases"] == 2
    assert summary["overall_pass_rate"] == 0.5
    assert summary["summary_faithfulness_pass_rate"] == 0.5
    assert summary["question_quality_pass_rate"] == 0.5
    assert summary["total_medication_checks"] == 2
    assert summary["medication_support_pass_rate"] == 0.5
    assert summary["medication_safety_pass_rate"] == 1.0
    assert summary["unsupported_claim_count"] == 1
    assert summary["cases_with_issues"] == ["fail-case"]
