from __future__ import annotations

from typing import Iterable

from app.config import Settings
from app.evals.generation_judge import GenerationJudge, GenerationJudgeResult
from app.schemas import AnalysisResponse, JudgeMeta, MedicationResult


class RuntimeJudgeService:
    SAFE_SUMMARY = (
        "MedSpeak processed this report, but the draft explanation did not fully pass automated review. "
        "Only sections that passed review are shown below, and a clinician should interpret them in context."
    )
    SAFE_WARNING = "Some generated wording was withheld because it did not pass automated review."
    SAFE_QUESTION = "What are the most important next steps or follow-up questions from this report?"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.enabled = settings.runtime_judge_enabled
        self.fail_closed = settings.runtime_judge_fail_closed
        self.model = settings.runtime_judge_model
        self.judge = GenerationJudge.from_settings(settings, model=settings.runtime_judge_model) if self.enabled else None

    async def review(self, *, report_text: str, analysis: AnalysisResponse) -> AnalysisResponse:
        if not self.enabled:
            return self._with_judge_meta(
                analysis,
                JudgeMeta(status="skipped", model=self.model, issues=["Runtime judge is disabled."]),
            )

        if self.judge is None or self.judge.client is None:
            return self._with_judge_meta(
                analysis,
                JudgeMeta(status="skipped", model=self.model, issues=["Runtime judge skipped because OpenAI is not configured."]),
            )

        try:
            judge_result = await self.judge.evaluate(report_text=report_text, analysis=analysis)
        except Exception as exc:
            judge_meta = JudgeMeta(
                status="unavailable",
                model=self.model,
                issues=[f"Runtime judge was unavailable: {exc}"],
                blocked_sections=["summary", "warnings", "questions_for_doctor"] if self.fail_closed else [],
            )
            return self._sanitize_analysis(analysis, judge_meta, judge_result=None) if self.fail_closed else self._with_judge_meta(analysis, judge_meta)

        if judge_result.overall_passed:
            return self._with_judge_meta(
                analysis,
                JudgeMeta(status="passed", model=self.model),
            )

        judge_meta = JudgeMeta(
            status="failed",
            model=self.model,
            issues=self._collect_issues(judge_result),
            blocked_sections=self._blocked_sections(judge_result),
        )
        return self._sanitize_analysis(analysis, judge_meta, judge_result=judge_result)

    def _sanitize_analysis(
        self,
        analysis: AnalysisResponse,
        judge_meta: JudgeMeta,
        *,
        judge_result: GenerationJudgeResult | None,
    ) -> AnalysisResponse:
        allowed_medications = self._allowed_medications(analysis.medications, judge_result)
        warnings = [self.SAFE_WARNING]
        if allowed_medications:
            warnings.append("Only medication entries that passed the automated review are shown below.")
        blocked_sections = set(judge_meta.blocked_sections)
        partial_reasons = list(analysis.meta.partial_data_reasons)
        partial_reasons.append("The draft explanation did not pass runtime judge review. MedSpeak returned a reduced safe response instead.")
        partial_reasons.extend(issue for issue in judge_meta.issues if issue not in partial_reasons)

        questions = [self.SAFE_QUESTION]
        if allowed_medications:
            questions.extend(
                f"How does {medication.name} fit with the rest of this report, and do I need any monitoring?"
                for medication in allowed_medications
                if medication.status == "current"
            )

        safe_summary = self.SAFE_SUMMARY if {"summary", "warnings", "questions_for_doctor"} & blocked_sections else analysis.summary

        return analysis.model_copy(
            update={
                "summary": safe_summary,
                "warnings": self._dedupe(warnings),
                "labs": [],
                "questions_for_doctor": self._dedupe(questions)[:5],
                "medications": allowed_medications,
                "diagnoses": [],
                "vitals": [],
                "allergies": [],
                "surgeries": [],
                "risk_factors": [],
                "meta": analysis.meta.model_copy(
                    update={
                        "partial_data": True,
                        "partial_data_reasons": self._dedupe(partial_reasons),
                        "fallback_used": True,
                        "judge": judge_meta,
                    }
                ),
            }
        )

    def _allowed_medications(
        self,
        medications: list[MedicationResult],
        judge_result: GenerationJudgeResult | None,
    ) -> list[MedicationResult]:
        if judge_result is None:
            return []
        safe_names = {
            self._normalize_name(check.medication)
            for check in judge_result.medication_checks
            if check.supported and check.safe
        }
        return [medication for medication in medications if self._normalize_name(medication.name) in safe_names]

    def _blocked_sections(self, judge_result: GenerationJudgeResult) -> list[str]:
        blocked = set()
        if not judge_result.overall_passed:
            blocked.update({"labs", "diagnoses", "vitals", "allergies", "surgeries", "risk_factors"})
        if not judge_result.summary_faithfulness.passed:
            blocked.add("summary")
        if not judge_result.safety_check.passed:
            blocked.update({"summary", "warnings", "questions_for_doctor"})
        if not judge_result.question_quality.passed:
            blocked.add("questions_for_doctor")
        if any(not (check.supported and check.safe) for check in judge_result.medication_checks):
            blocked.add("medications")
        return sorted(blocked)

    def _collect_issues(self, judge_result: GenerationJudgeResult) -> list[str]:
        issues: list[str] = []
        issues.extend(judge_result.summary_faithfulness.issues)
        issues.extend(judge_result.safety_check.issues)
        issues.extend(judge_result.question_quality.issues)
        for check in judge_result.medication_checks:
            issues.extend(check.issues)
        issues.extend(judge_result.unsupported_claims)
        if not issues:
            issues.append("The response did not pass automated review.")
        return self._dedupe(issues)

    def _with_judge_meta(self, analysis: AnalysisResponse, judge_meta: JudgeMeta) -> AnalysisResponse:
        return analysis.model_copy(update={"meta": analysis.meta.model_copy(update={"judge": judge_meta})})

    def _normalize_name(self, value: str) -> str:
        return " ".join(value.lower().replace("(", " ").replace(")", " ").split())

    def _dedupe(self, items: Iterable[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            cleaned = str(item).strip()
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            unique.append(cleaned)
        return unique
