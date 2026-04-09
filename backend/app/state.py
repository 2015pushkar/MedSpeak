from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.schemas import DiagnosisResult, DocumentType, LabResult, MedicationResult


class PipelineState(TypedDict, total=False):
    input_text: str
    document_type: DocumentType
    agent_targets: list[str]
    summary: str
    warnings: Annotated[list[str], operator.add]
    questions_for_doctor: list[str]
    labs: list[LabResult]
    medications: list[MedicationResult]
    diagnoses: list[DiagnosisResult]
    partial_data_reasons: Annotated[list[str], operator.add]
    fallback_used: Annotated[bool, operator.or_]
    sources: Annotated[list[str], operator.add]

