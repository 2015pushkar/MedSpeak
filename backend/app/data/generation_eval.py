from __future__ import annotations

from dataclasses import dataclass

from app.data.sample_reports import LAB_REPORT_TEXT, MEDICATION_REPORT_TEXT, MIXED_REPORT_TEXT, PAMELA_ROGERS_HP_TEXT, UNKNOWN_REPORT_TEXT


@dataclass(frozen=True)
class GenerationEvalCase:
    name: str
    report_text: str
    description: str


GENERATION_EVAL_CASES: tuple[GenerationEvalCase, ...] = (
    GenerationEvalCase(
        name="lab-basic",
        report_text=LAB_REPORT_TEXT,
        description="Simple lab report with out-of-range glucose and potassium values.",
    ),
    GenerationEvalCase(
        name="medication-list",
        report_text=MEDICATION_REPORT_TEXT,
        description="Basic medication list for grounding and summary checks.",
    ),
    GenerationEvalCase(
        name="mixed-discharge",
        report_text=MIXED_REPORT_TEXT,
        description="Mixed discharge summary with diagnoses, labs, and medications.",
    ),
    GenerationEvalCase(
        name="history-and-physical",
        report_text=PAMELA_ROGERS_HP_TEXT,
        description="History and physical document with active problems, vitals, surgeries, allergies, and risk factors.",
    ),
    GenerationEvalCase(
        name="unknown-message",
        report_text=UNKNOWN_REPORT_TEXT,
        description="Low-structure portal message to test restraint and unknown-format handling.",
    ),
)


def select_generation_eval_cases(names: list[str] | None = None) -> tuple[GenerationEvalCase, ...]:
    if not names:
        return GENERATION_EVAL_CASES
    lookup = {case.name: case for case in GENERATION_EVAL_CASES}
    selected: list[GenerationEvalCase] = []
    for name in names:
        if name not in lookup:
            raise ValueError(f"Unknown generation eval case: {name}")
        selected.append(lookup[name])
    return tuple(selected)
