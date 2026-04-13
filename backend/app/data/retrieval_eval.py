from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalEvalCase:
    query: str
    expected_medication: str
    source_document: str


RETRIEVAL_EVAL_CASES: tuple[RetrievalEvalCase, ...] = (
    RetrievalEvalCase("Lisinopril", "Lisinopril", "discharge medications"),
    RetrievalEvalCase("Prinivil", "Lisinopril", "brand alias"),
    RetrievalEvalCase("Metformin", "Metformin", "lab follow-up plan"),
    RetrievalEvalCase("Lipitor", "Atorvastatin", "brand alias"),
    RetrievalEvalCase("Levothyroxine", "Levothyroxine", "thyroid refill"),
    RetrievalEvalCase("Norvasc", "Amlodipine", "brand alias"),
    RetrievalEvalCase("Prilosec", "Omeprazole", "brand alias"),
    RetrievalEvalCase("Losartan", "Losartan", "blood pressure summary"),
    RetrievalEvalCase("HCTZ", "Hydrochlorothiazide", "short alias"),
    RetrievalEvalCase("Sertraline", "Sertraline", "medication list"),
)
