from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.schemas import (
    AllergyResult,
    DiagnosisResult,
    DocumentType,
    LabResult,
    MedicationResult,
    RiskFactorResult,
    SurgeryResult,
    VitalResult,
)


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
    vitals: list[VitalResult]
    allergies: list[AllergyResult]
    surgeries: list[SurgeryResult]
    risk_factors: list[RiskFactorResult]
    partial_data_reasons: Annotated[list[str], operator.add]
    fallback_used: Annotated[bool, operator.or_]
    sources: Annotated[list[str], operator.add]
    processing_trace: Annotated[dict[str, str], operator.or_]
