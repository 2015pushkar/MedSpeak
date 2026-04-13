from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentType = Literal["lab", "medication", "diagnosis", "mixed", "unknown"]
LabStatus = Literal["low", "normal", "high", "unknown"]
GroundingStatus = Literal["rag", "openfda_live", "text_only"]


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Any | None = None


class LabResult(BaseModel):
    name: str
    value: str
    unit: str
    reference_range: str
    status: LabStatus
    explanation: str


class MedicationEvidence(BaseModel):
    source: str
    label_section: str
    chunk_id: str
    snippet: str


class MedicationResult(BaseModel):
    name: str
    purpose: str
    common_side_effects: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    fda_enriched: bool = False
    grounding_status: GroundingStatus = "text_only"
    evidence: list[MedicationEvidence] = Field(default_factory=list)


class DiagnosisResult(BaseModel):
    term: str
    plain_language: str


class AnalysisMeta(BaseModel):
    rate_limit_remaining: int
    daily_limit: int
    rate_limit_reset_at: str
    partial_data: bool = False
    partial_data_reasons: list[str] = Field(default_factory=list)
    fallback_used: bool = False
    sources: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    document_type: DocumentType
    summary: str
    warnings: list[str] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[MedicationResult] = Field(default_factory=list)
    diagnoses: list[DiagnosisResult] = Field(default_factory=list)
    questions_for_doctor: list[str] = Field(default_factory=list)
    disclaimer: str
    meta: AnalysisMeta


class RateStatusResponse(BaseModel):
    remaining: int
    daily_limit: int
    reset_at: str


class DependencyHealth(BaseModel):
    status: str
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, DependencyHealth]
