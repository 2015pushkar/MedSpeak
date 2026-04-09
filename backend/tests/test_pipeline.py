from __future__ import annotations

from app.config import Settings
from app.pipeline import MedicalPipeline
from app.tools.openfda import MedicationEnrichment
from tests.sample_reports import (
    DIAGNOSIS_REPORT_TEXT,
    LAB_REPORT_TEXT,
    MEDICATION_REPORT_TEXT,
    MIXED_REPORT_TEXT,
    UNKNOWN_REPORT_TEXT,
)


class DummyOpenFDA:
    async def lookup(self, medication_name: str) -> MedicationEnrichment:
        return MedicationEnrichment(
            purpose=f"{medication_name} label purpose.",
            common_side_effects=["Dizziness."],
            cautions=["Monitor as directed."],
            fda_enriched=True,
        )


class FailingOpenFDA:
    async def lookup(self, _: str) -> MedicationEnrichment:
        raise RuntimeError("timeout")


def build_pipeline(openfda_tool) -> MedicalPipeline:
    settings = Settings(openai_api_key=None)
    return MedicalPipeline(settings, openfda_tool)


async def test_pipeline_lab_only():
    pipeline = build_pipeline(DummyOpenFDA())
    result = await pipeline.analyze(
        text=LAB_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.document_type == "lab"
    assert result.labs
    assert result.meta.fallback_used is True


async def test_pipeline_medication_only():
    pipeline = build_pipeline(DummyOpenFDA())
    result = await pipeline.analyze(
        text=MEDICATION_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.document_type == "medication"
    assert result.medications
    assert all(item.fda_enriched for item in result.medications)


async def test_pipeline_mixed_content():
    pipeline = build_pipeline(DummyOpenFDA())
    result = await pipeline.analyze(
        text=MIXED_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.document_type == "mixed"
    assert result.labs
    assert result.medications
    assert result.diagnoses


async def test_pipeline_unknown_content():
    pipeline = build_pipeline(DummyOpenFDA())
    result = await pipeline.analyze(
        text=UNKNOWN_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.document_type == "unknown"
    assert result.warnings


async def test_pipeline_openfda_timeout_sets_partial_data():
    pipeline = build_pipeline(FailingOpenFDA())
    result = await pipeline.analyze(
        text=MEDICATION_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.document_type == "medication"
    assert result.meta.partial_data is True
    assert "OpenFDA enrichment was unavailable" in result.meta.partial_data_reasons[0]


async def test_pipeline_diagnosis_only():
    pipeline = build_pipeline(DummyOpenFDA())
    result = await pipeline.analyze(
        text=DIAGNOSIS_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.document_type == "diagnosis"
    assert result.diagnoses

