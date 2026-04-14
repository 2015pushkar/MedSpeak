from __future__ import annotations

from app.config import Settings
from app.evals.generation_judge import GenerationJudgeResult, JudgeBinaryCheck, MedicationJudgeCheck
from app.schemas import (
    AnalysisMeta,
    AnalysisResponse,
    DiagnosisResult,
    MedicationResult,
    ProcessingTrace,
)
from app.services.runtime_judge import RuntimeJudgeService


class PassingJudge:
    client = object()

    async def evaluate(self, *, report_text: str, analysis: AnalysisResponse):
        return GenerationJudgeResult(
            summary_faithfulness=JudgeBinaryCheck(passed=True),
            medication_checks=[
                MedicationJudgeCheck(
                    medication=medication.name,
                    grounding_status=medication.grounding_status,
                    supported=True,
                    safe=True,
                )
                for medication in analysis.medications
            ],
            safety_check=JudgeBinaryCheck(passed=True),
            question_quality=JudgeBinaryCheck(passed=True),
            unsupported_claims=[],
            overall_passed=True,
        )


class FailingJudge:
    client = object()

    async def evaluate(self, *, report_text: str, analysis: AnalysisResponse):
        return GenerationJudgeResult(
            summary_faithfulness=JudgeBinaryCheck(passed=False, issues=["Summary added unsupported wording."]),
            medication_checks=[
                MedicationJudgeCheck(
                    medication=analysis.medications[0].name,
                    grounding_status=analysis.medications[0].grounding_status,
                    supported=False,
                    safe=False,
                    issues=["Medication explanation was not adequately supported."],
                )
            ],
            safety_check=JudgeBinaryCheck(passed=False, issues=["The draft used unsafe treatment-like wording."]),
            question_quality=JudgeBinaryCheck(passed=False, issues=["Questions were too specific for the evidence."]),
            unsupported_claims=["The summary implied a diagnosis not stated in the report."],
            overall_passed=False,
        )


def build_analysis() -> AnalysisResponse:
    return AnalysisResponse(
        document_type="mixed",
        summary="This summary should be replaced when the judge fails.",
        warnings=["Original warning."],
        labs=[],
        medications=[
            MedicationResult(
                name="Lisinopril",
                purpose="Lisinopril is used to treat hypertension.",
                common_side_effects=[],
                cautions=["Monitor blood pressure."],
                fda_enriched=True,
                grounding_status="rag",
                status="current",
                grounding_note="grounded from local corpus",
                evidence=[],
            )
        ],
        diagnoses=[DiagnosisResult(term="Hypertension", plain_language="High blood pressure.")],
        vitals=[],
        allergies=[],
        surgeries=[],
        risk_factors=[],
        questions_for_doctor=["Original question?"],
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


async def test_runtime_judge_passes_response_through_when_passed():
    service = RuntimeJudgeService(Settings(runtime_judge_enabled=True, openai_api_key=None))
    service.judge = PassingJudge()

    result = await service.review(report_text="Report text", analysis=build_analysis())

    assert result.summary == "This summary should be replaced when the judge fails."
    assert result.meta.judge.status == "passed"
    assert result.meta.fallback_used is False


async def test_runtime_judge_fails_closed_and_reduces_response():
    service = RuntimeJudgeService(Settings(runtime_judge_enabled=True, openai_api_key=None))
    service.judge = FailingJudge()

    result = await service.review(report_text="Report text", analysis=build_analysis())

    assert "did not fully pass automated review" in result.summary.lower()
    assert result.questions_for_doctor == [service.SAFE_QUESTION]
    assert result.medications == []
    assert result.labs == []
    assert result.diagnoses == []
    assert result.vitals == []
    assert result.allergies == []
    assert result.surgeries == []
    assert result.risk_factors == []
    assert result.meta.partial_data is True
    assert result.meta.fallback_used is True
    assert result.meta.judge.status == "failed"
    assert "diagnoses" in result.meta.judge.blocked_sections
    assert "summary added unsupported wording." in " ".join(result.meta.judge.issues).lower()
