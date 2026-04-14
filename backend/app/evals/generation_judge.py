from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

from app.config import Settings
from app.prompts.eval_prompts import GENERATION_JUDGE_PROMPT
from app.schemas import AnalysisResponse


class JudgeBinaryCheck(BaseModel):
    passed: bool = False
    issues: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)


class MedicationJudgeCheck(BaseModel):
    medication: str
    grounding_status: str = "text_only"
    supported: bool = False
    safe: bool = False
    issues: list[str] = Field(default_factory=list)
    evidence_used: list[str] = Field(default_factory=list)


class GenerationJudgeResult(BaseModel):
    summary_faithfulness: JudgeBinaryCheck = Field(default_factory=JudgeBinaryCheck)
    medication_checks: list[MedicationJudgeCheck] = Field(default_factory=list)
    safety_check: JudgeBinaryCheck = Field(default_factory=JudgeBinaryCheck)
    question_quality: JudgeBinaryCheck = Field(default_factory=JudgeBinaryCheck)
    unsupported_claims: list[str] = Field(default_factory=list)
    overall_passed: bool = False


@dataclass(frozen=True)
class GenerationEvaluationRecord:
    case_name: str
    analysis: AnalysisResponse
    judge_result: GenerationJudgeResult
    description: str = ""
    raw_synthesizer_response: dict[str, Any] | None = None
    raw_judge_response: dict[str, Any] | None = None


def build_generation_judge_payload(*, report_text: str, analysis: AnalysisResponse, case_name: str | None = None) -> dict[str, Any]:
    return {
        "case_name": case_name or "unnamed-case",
        "original_report": report_text,
        "medspeak_output": {
            "document_type": analysis.document_type,
            "summary": analysis.summary,
            "warnings": analysis.warnings,
            "labs": [lab.model_dump() for lab in analysis.labs],
            "medications": [medication.model_dump() for medication in analysis.medications],
            "diagnoses": [diagnosis.model_dump() for diagnosis in analysis.diagnoses],
            "vitals": [vital.model_dump() for vital in analysis.vitals],
            "allergies": [allergy.model_dump() for allergy in analysis.allergies],
            "surgeries": [surgery.model_dump() for surgery in analysis.surgeries],
            "risk_factors": [factor.model_dump() for factor in analysis.risk_factors],
            "questions_for_doctor": analysis.questions_for_doctor,
            "disclaimer": analysis.disclaimer,
            "meta": {
                "partial_data": analysis.meta.partial_data,
                "partial_data_reasons": analysis.meta.partial_data_reasons,
                "fallback_used": analysis.meta.fallback_used,
                "sources": analysis.meta.sources,
                "processing_trace": analysis.meta.processing_trace.model_dump(),
            },
        },
    }


def summarize_generation_evaluations(records: list[GenerationEvaluationRecord]) -> dict[str, Any]:
    total_cases = len(records)
    medications = [check for record in records for check in record.judge_result.medication_checks]
    summary_passes = sum(record.judge_result.summary_faithfulness.passed for record in records)
    safety_passes = sum(record.judge_result.safety_check.passed for record in records)
    question_passes = sum(record.judge_result.question_quality.passed for record in records)
    overall_passes = sum(record.judge_result.overall_passed for record in records)
    medication_support_passes = sum(check.supported for check in medications)
    medication_safety_passes = sum(check.safe for check in medications)
    unsupported_claim_count = sum(len(record.judge_result.unsupported_claims) for record in records)
    cases_with_issues = [record.case_name for record in records if not record.judge_result.overall_passed]

    return {
        "total_cases": total_cases,
        "overall_pass_rate": round(overall_passes / total_cases, 3) if total_cases else 0.0,
        "summary_faithfulness_pass_rate": round(summary_passes / total_cases, 3) if total_cases else 0.0,
        "safety_pass_rate": round(safety_passes / total_cases, 3) if total_cases else 0.0,
        "question_quality_pass_rate": round(question_passes / total_cases, 3) if total_cases else 0.0,
        "total_medication_checks": len(medications),
        "medication_support_pass_rate": round(medication_support_passes / len(medications), 3) if medications else 0.0,
        "medication_safety_pass_rate": round(medication_safety_passes / len(medications), 3) if medications else 0.0,
        "unsupported_claim_count": unsupported_claim_count,
        "cases_with_issues": cases_with_issues,
    }


class GenerationJudge:
    def __init__(
        self,
        *,
        client: AsyncOpenAI | None,
        model: str,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 0.5,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.last_raw_response: dict[str, Any] | None = None

    @classmethod
    def from_settings(cls, settings: Settings, *, model: str = "gpt-4o-mini") -> "GenerationJudge":
        client = None
        if settings.openai_enabled:
            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
            )
        return cls(
            client=client,
            model=model,
            max_retries=settings.llm_max_retries,
            retry_base_delay_seconds=settings.llm_retry_base_delay_seconds,
        )

    async def evaluate(
        self,
        *,
        report_text: str,
        analysis: AnalysisResponse,
        case_name: str | None = None,
    ) -> GenerationJudgeResult:
        if self.client is None:
            raise RuntimeError("LLM-as-judge evaluation requires OPENAI_API_KEY to be configured.")

        self.last_raw_response = None
        payload = build_generation_judge_payload(report_text=report_text, analysis=analysis, case_name=case_name)
        completion = await self._json_completion(json.dumps(payload, ensure_ascii=True))
        self.last_raw_response = completion
        try:
            result = GenerationJudgeResult.model_validate(completion)
        except ValidationError as exc:
            raise RuntimeError(f"Judge model returned an invalid evaluation payload: {exc}") from exc
        return self._normalize_result(result, analysis)

    async def _json_completion(self, user_content: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    response_format={"type": "json_object"},
                    temperature=0,
                    messages=[
                        {"role": "system", "content": GENERATION_JUDGE_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                )
                content = response.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:  # pragma: no cover - network/model behavior
                last_error = exc
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_base_delay_seconds * (2**attempt))
        raise RuntimeError(f"Judge model was unavailable after retries: {last_error}") from last_error

    def _normalize_result(self, result: GenerationJudgeResult, analysis: AnalysisResponse) -> GenerationJudgeResult:
        keyed_checks = {
            self._normalize_medication_name(check.medication): check
            for check in result.medication_checks
            if check.medication.strip()
        }
        normalized_checks: list[MedicationJudgeCheck] = []

        for medication in analysis.medications:
            key = self._normalize_medication_name(medication.name)
            matched = keyed_checks.get(key)
            if matched:
                normalized_checks.append(
                    matched.model_copy(
                        update={
                            "medication": medication.name,
                            "grounding_status": medication.grounding_status,
                        }
                    )
                )
                continue
            normalized_checks.append(
                MedicationJudgeCheck(
                    medication=medication.name,
                    grounding_status=medication.grounding_status,
                    supported=False,
                    safe=False,
                    issues=["Judge output omitted this medication from evaluation."],
                    evidence_used=[],
                )
            )

        overall_passed = (
            result.summary_faithfulness.passed
            and result.safety_check.passed
            and result.question_quality.passed
            and all(check.supported and check.safe for check in normalized_checks)
            and not result.unsupported_claims
        )

        return result.model_copy(
            update={
                "medication_checks": normalized_checks,
                "overall_passed": overall_passed,
            }
        )

    def _normalize_medication_name(self, value: str) -> str:
        lowered = re.sub(r"\([^)]*\)", "", value.lower())
        return re.sub(r"\s+", " ", lowered).strip()
