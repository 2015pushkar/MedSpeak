from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from app.prompts.agent_prompts import SAFETY_REWRITE_PROMPT


@dataclass(frozen=True)
class SafetyReviewResult:
    summary: str
    warnings: list[str]
    questions_for_doctor: list[str]
    partial_data_reasons: list[str]
    rewrite_used: bool = False
    canned_response_used: bool = False


class SafetyService:
    SAFE_SUMMARY = (
        "MedSpeak could not safely produce a fuller educational explanation from this document without risking "
        "overly specific medical guidance. Review the extracted findings with your clinician."
    )
    SAFE_WARNING = "Some model-generated wording was replaced after safety review to avoid diagnosis or treatment advice."
    SAFE_QUESTION = "What are the most important next steps or follow-up questions from this report?"

    def __init__(self) -> None:
        self.patterns: dict[str, re.Pattern[str]] = {
            "dosage": re.compile(r"\b(?:take|start|stop|increase|decrease|adjust)\b.{0,25}\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|units?)\b", re.IGNORECASE),
            "treatment": re.compile(r"\b(?:you should|need to|must|start taking|stop taking|change your medication|treat this with)\b", re.IGNORECASE),
            "diagnosis": re.compile(r"\b(?:this means you have|this confirms|you have been diagnosed with|definitely indicates)\b", re.IGNORECASE),
        }

    def violations(self, summary: str, warnings: list[str], questions_for_doctor: list[str]) -> list[str]:
        text = " ".join([summary, *warnings, *questions_for_doctor])
        hits: list[str] = []
        for label, pattern in self.patterns.items():
            if pattern.search(text):
                hits.append(label)
        return hits

    async def enforce(
        self,
        *,
        summary: str,
        warnings: list[str],
        questions_for_doctor: list[str],
        client: AsyncOpenAI | None,
        model: str,
    ) -> SafetyReviewResult:
        detected = self.violations(summary, warnings, questions_for_doctor)
        if not detected:
            return SafetyReviewResult(
                summary=summary,
                warnings=warnings,
                questions_for_doctor=questions_for_doctor,
                partial_data_reasons=[],
            )

        if client:
            rewritten = await self._rewrite(
                client=client,
                model=model,
                summary=summary,
                warnings=warnings,
                questions_for_doctor=questions_for_doctor,
            )
            if rewritten:
                next_violations = self.violations(
                    rewritten["summary"],
                    rewritten["warnings"],
                    rewritten["questions_for_doctor"],
                )
                if not next_violations:
                    return SafetyReviewResult(
                        summary=rewritten["summary"],
                        warnings=self._dedupe([*rewritten["warnings"], self.SAFE_WARNING]),
                        questions_for_doctor=rewritten["questions_for_doctor"],
                        partial_data_reasons=["A safety rewrite was applied to remove diagnosis or treatment-like language."],
                        rewrite_used=True,
                    )

        return SafetyReviewResult(
            summary=self.SAFE_SUMMARY,
            warnings=self._dedupe([*warnings, self.SAFE_WARNING]),
            questions_for_doctor=questions_for_doctor or [self.SAFE_QUESTION],
            partial_data_reasons=["A canned safe response replaced an unsafe model draft."],
            rewrite_used=client is not None,
            canned_response_used=True,
        )

    async def _rewrite(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        summary: str,
        warnings: list[str],
        questions_for_doctor: list[str],
    ) -> dict[str, list[str] | str] | None:
        try:
            response = await client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SAFETY_REWRITE_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "summary": summary,
                                "warnings": warnings,
                                "questions_for_doctor": questions_for_doctor,
                            }
                        ),
                    },
                ],
            )
        except Exception:
            return None
        content = response.choices[0].message.content
        if not content:
            return None
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        next_summary = str(payload.get("summary", "")).strip()
        next_warnings = [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()]
        next_questions = [str(item).strip() for item in payload.get("questions_for_doctor", []) if str(item).strip()]
        if not next_summary:
            return None
        return {
            "summary": next_summary,
            "warnings": next_warnings,
            "questions_for_doctor": next_questions or [self.SAFE_QUESTION],
        }

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            normalized = value.lower().strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value.strip())
        return unique
