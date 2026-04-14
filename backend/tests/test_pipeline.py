from __future__ import annotations

from app.config import Settings
from app.pipeline import MedicalPipeline
from app.services.medication_rag import (
    DeterministicEmbedder,
    InMemoryMedicationVectorStore,
    MedicationRAGService,
    build_seed_alias_map,
)
from app.tools.openfda import MedicationEnrichment, MedicationLabelDocument
from tests.sample_reports import (
    DIAGNOSIS_REPORT_TEXT,
    LAB_REPORT_TEXT,
    MEDICATION_REPORT_TEXT,
    MIXED_REPORT_TEXT,
    PAMELA_ROGERS_HP_TEXT,
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

    async def fetch_label_document(self, medication_name: str, *, aliases: tuple[str, ...] = ()) -> MedicationLabelDocument | None:
        return MedicationLabelDocument(
            canonical_name=medication_name.strip().title(),
            aliases=aliases or (medication_name.strip().title(),),
            sections={
                "indications_and_usage": f"{medication_name} is used in the label indications section.",
                "warnings_and_cautions": f"{medication_name} has warning language in the FDA label.",
                "adverse_reactions": f"{medication_name} may cause dizziness and nausea.",
            },
        )


class FailingOpenFDA:
    async def lookup(self, _: str) -> MedicationEnrichment:
        raise RuntimeError("timeout")

    async def fetch_label_document(self, _: str, *, aliases: tuple[str, ...] = ()) -> MedicationLabelDocument | None:
        raise RuntimeError("timeout")


def build_pipeline(openfda_tool) -> MedicalPipeline:
    settings = Settings(openai_api_key=None)
    pipeline = MedicalPipeline(settings, openfda_tool)
    pipeline.medication_rag = MedicationRAGService(
        store=InMemoryMedicationVectorStore(),
        embedder=DeterministicEmbedder(),
        alias_map=build_seed_alias_map(),
    )
    return pipeline


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
    assert all(item.grounding_status == "openfda_live" for item in result.medications)


async def test_pipeline_promotes_live_openfda_medication_into_local_rag_for_next_request():
    pipeline = build_pipeline(DummyOpenFDA())
    text = "Discharge Medications:\nTylenol 500 mg as needed"

    first = await pipeline.analyze(
        text=text,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert first.medications
    assert first.medications[0].grounding_status == "openfda_live"
    assert "saved for future local grounding" in first.medications[0].grounding_note.lower()

    second = await pipeline.analyze(
        text=text,
        rate_limit_remaining=3,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert second.medications
    assert second.medications[0].grounding_status == "rag"


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
    assert result.meta.processing_trace.classifier in {"heuristic", "llm"}


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
    assert any("OpenFDA enrichment was unavailable" in reason for reason in result.meta.partial_data_reasons)
    assert all(item.grounding_status == "text_only" for item in result.medications)


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


async def test_pipeline_history_and_physical_sections():
    pipeline = build_pipeline(DummyOpenFDA())
    result = await pipeline.analyze(
        text=PAMELA_ROGERS_HP_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )

    assert result.document_type == "mixed"
    assert {item.term.lower() for item in result.diagnoses} >= {"chest pain", "shortness of breath", "hypertension"}
    assert all("hysterectomy" not in item.term.lower() for item in result.diagnoses)
    assert result.allergies
    assert result.allergies[0].substance.lower() == "penicillin"
    assert "rash and hives" in result.allergies[0].reaction.lower()
    assert any("hysterectomy" in item.procedure.lower() for item in result.surgeries)
    assert any("premature cad" in item.factor.lower() for item in result.risk_factors)
    vital_values = {item.name: item.value for item in result.vitals}
    assert vital_values["Blood Pressure"] == "168/98"
    assert vital_values["Pulse"] == "90"
    assert vital_values["Respirations"] == "20"
    assert vital_values["Temperature"] == "37"
    assert all(item.grounding_status != "rag" for item in result.medications)
    assert all(item.status in {"historical", "otc_prn"} for item in result.medications)
    assert "taking three medications" not in result.summary.lower()
    joined_questions = " ".join(result.questions_for_doctor).lower()
    assert "chest pain" in joined_questions
    assert "cimetidine" not in joined_questions
