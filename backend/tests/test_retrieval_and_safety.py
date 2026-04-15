from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.pipeline import MedicalPipeline
from app.services.medication_rag import (
    DeterministicEmbedder,
    InMemoryMedicationVectorStore,
    MedicationRAGService,
    build_seed_alias_map,
)
from app.services.safety import SafetyService
from app.tools.openfda import MedicationEnrichment, MedicationLabelDocument
from tests.sample_reports import MEDICATION_REPORT_TEXT, PAMELA_ROGERS_HP_TEXT


class DummyOpenFDA:
    async def lookup(self, medication_name: str) -> MedicationEnrichment:
        return MedicationEnrichment(
            purpose=f"{medication_name} label purpose.",
            common_side_effects=["Dizziness."],
            cautions=["Monitor as directed."],
            fda_enriched=True,
        )


class FailingCompletions:
    async def create(self, **_: object) -> None:
        raise RuntimeError("boom")


def build_rag_service() -> MedicationRAGService:
    return MedicationRAGService(
        store=InMemoryMedicationVectorStore(),
        embedder=DeterministicEmbedder(),
        alias_map=build_seed_alias_map(),
    )


async def seed_lisinopril(rag_service: MedicationRAGService) -> None:
    await rag_service.ingest_documents(
        [
            MedicationLabelDocument(
                canonical_name="Lisinopril",
                aliases=("Prinivil", "Zestril"),
                sections={
                    "indications_and_usage": "Lisinopril is indicated for the treatment of hypertension in adults and children.",
                    "adverse_reactions": "Common adverse reactions include dizziness and headache.",
                    "warnings_and_cautions": "Monitor blood pressure and kidney function during therapy.",
                },
            )
        ]
    )


async def seed_ibuprofen_and_percocet(rag_service: MedicationRAGService) -> None:
    await rag_service.ingest_documents(
        [
            MedicationLabelDocument(
                canonical_name="Ibuprofen",
                aliases=("Advil", "Motrin"),
                sections={
                    "indications_and_usage": "Ibuprofen temporarily relieves minor aches and pains due to headache and backache and reduces fever.",
                    "warnings_and_cautions": "Ibuprofen may cause severe stomach bleeding and allergic reactions in some patients.",
                    "adverse_reactions": "Common side effects may include nausea and stomach upset.",
                },
            ),
            MedicationLabelDocument(
                canonical_name="Percocet",
                aliases=("Oxycodone Acetaminophen", "Oxycodone/Acetaminophen", "Oxycodone and Acetaminophen"),
                sections={
                    "indications_and_usage": "Percocet is indicated for the management of pain severe enough to require an opioid analgesic when alternative treatments are inadequate.",
                    "warnings_and_cautions": "Percocet exposes patients to risks of addiction, abuse, misuse, respiratory depression, and overdose.",
                    "adverse_reactions": "Common adverse reactions include nausea, sedation, constipation, and dizziness.",
                },
            ),
        ]
    )


async def test_rag_service_retrieves_seed_medication():
    rag_service = build_rag_service()
    await seed_lisinopril(rag_service)

    retrieval = await rag_service.retrieve("Prinivil", top_k=3)
    assert retrieval.chunks
    assert retrieval.chunks[0].canonical_name == "Lisinopril"

    grounding = await rag_service.ground_medication("Lisinopril", "Fallback purpose.", top_k=3)
    assert grounding is not None
    assert "hypertension" in grounding.purpose.lower()
    assert grounding.evidence


async def test_rag_service_skips_reembedding_unchanged_documents():
    rag_service = build_rag_service()
    document = MedicationLabelDocument(
        canonical_name="Lisinopril",
        aliases=("Prinivil", "Zestril"),
        sections={
            "indications_and_usage": "Lisinopril is indicated for the treatment of hypertension in adults and children.",
            "adverse_reactions": "Common adverse reactions include dizziness and headache.",
            "warnings_and_cautions": "Monitor blood pressure and kidney function during therapy.",
        },
    )

    first = await rag_service.ingest_documents([document])
    second = await rag_service.ingest_documents([document])

    assert first > 0
    assert second == 0


async def test_rag_service_resolves_ibuprofen_and_percocet_aliases():
    rag_service = build_rag_service()
    await seed_ibuprofen_and_percocet(rag_service)

    ibuprofen_retrieval = await rag_service.retrieve("ibuprofen (Advil)", top_k=3)
    assert ibuprofen_retrieval.resolved_name == "Ibuprofen"
    assert ibuprofen_retrieval.chunks
    assert ibuprofen_retrieval.chunks[0].canonical_name == "Ibuprofen"

    percocet_retrieval = await rag_service.retrieve("oxycodone/acetaminophen", top_k=3)
    assert percocet_retrieval.resolved_name == "Percocet"
    assert percocet_retrieval.chunks
    assert percocet_retrieval.chunks[0].canonical_name == "Percocet"

    dosage_form_retrieval = await rag_service.retrieve("ibuprofen 600 mg tablet", top_k=3)
    assert dosage_form_retrieval.resolved_name == "Ibuprofen"
    assert dosage_form_retrieval.chunks
    assert dosage_form_retrieval.chunks[0].canonical_name == "Ibuprofen"


async def test_pipeline_prefers_rag_grounding_when_corpus_available():
    pipeline = MedicalPipeline(Settings(openai_api_key=None), DummyOpenFDA())
    pipeline.medication_rag = build_rag_service()
    await seed_lisinopril(pipeline.medication_rag)

    result = await pipeline.analyze(
        text=MEDICATION_REPORT_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )
    assert result.medications
    assert result.medications[0].grounding_status == "rag"
    assert result.medications[0].evidence
    assert "memory" in result.meta.sources


async def test_pipeline_does_not_ground_unknown_medications_to_seed_neighbors():
    pipeline = MedicalPipeline(Settings(openai_api_key=None), DummyOpenFDA())
    pipeline.medication_rag = build_rag_service()
    await seed_lisinopril(pipeline.medication_rag)

    result = await pipeline.analyze(
        text=PAMELA_ROGERS_HP_TEXT,
        rate_limit_remaining=4,
        daily_limit=5,
        reset_at="2099-01-01T00:00:00+00:00",
    )

    assert result.medications
    assert all(item.grounding_status != "rag" for item in result.medications)
    assert all("lisinopril" not in item.purpose.lower() for item in result.medications)
    assert all(item.grounding_status in {"openfda_live", "text_only"} for item in result.medications)
    await pipeline.medication_rag.wait_for_background_tasks()


async def test_safety_service_replaces_unsafe_language_without_client():
    service = SafetyService()
    result = await service.enforce(
        summary="You should take 10 mg of lisinopril because this confirms hypertension.",
        warnings=[],
        questions_for_doctor=[],
        client=None,
        model="unused",
    )
    assert result.canned_response_used is True
    assert "safe response" in result.partial_data_reasons[0].lower()
    assert result.warnings


async def test_json_completion_returns_failure_reason_after_retries():
    pipeline = MedicalPipeline(Settings(openai_api_key="test-key"), DummyOpenFDA())
    pipeline.client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
    completion = await pipeline._json_completion(
        model="gpt-4o-mini",
        prompt="Return JSON.",
        user_content="hello",
        failure_label="Test model",
    )
    assert completion.payload is None
    assert completion.failure_reason is not None
    assert "deterministic fallback" in completion.failure_reason.lower()
