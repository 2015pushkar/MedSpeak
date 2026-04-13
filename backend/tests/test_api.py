from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, pipeline, rate_limiter
from app.services.rate_limiter import InMemoryRateLimitStore
from app.tools.openfda import MedicationEnrichment
from tests.sample_reports import LAB_REPORT_TEXT, MIXED_REPORT_TEXT, build_pdf_bytes

client = TestClient(app)


class DummyOpenFDA:
    async def lookup(self, medication_name: str) -> MedicationEnrichment:
        return MedicationEnrichment(
            purpose=f"{medication_name} label purpose.",
            common_side_effects=["Dizziness."],
            cautions=["Monitor as directed."],
            fda_enriched=True,
        )


def setup_function() -> None:
    rate_limiter._store = InMemoryRateLimitStore()
    rate_limiter.daily_limit = 5
    pipeline.openfda_tool = DummyOpenFDA()


def test_analyze_raw_text_contract():
    response = client.post("/api/analyze", data={"raw_text": MIXED_REPORT_TEXT})
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_type"] == "mixed"
    assert "summary" in payload
    assert "meta" in payload
    assert isinstance(payload["labs"], list)
    assert isinstance(payload["medications"], list)
    assert isinstance(payload["diagnoses"], list)
    if payload["medications"]:
        assert "grounding_status" in payload["medications"][0]
        assert "evidence" in payload["medications"][0]


def test_analyze_pdf_contract():
    pdf_bytes = build_pdf_bytes(LAB_REPORT_TEXT)
    response = client.post(
        "/api/analyze",
        files={"file": ("lab_report.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["document_type"] == "lab"
    assert payload["labs"]


def test_missing_input_rejected():
    response = client.post("/api/analyze")
    assert response.status_code == 400
    assert response.json()["code"] == "missing_input"


def test_duplicate_input_rejected():
    pdf_bytes = build_pdf_bytes(LAB_REPORT_TEXT)
    response = client.post(
        "/api/analyze",
        data={"raw_text": LAB_REPORT_TEXT},
        files={"file": ("lab_report.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "multiple_inputs"


def test_unsupported_file_type_rejected():
    response = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_file_type"


def test_non_extractable_pdf_rejected():
    blank_pdf = build_pdf_bytes(None)
    response = client.post(
        "/api/analyze",
        files={"file": ("scan.pdf", blank_pdf, "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "non_extractable_pdf"


def test_rate_limit_exhaustion():
    rate_limiter.daily_limit = 1
    first = client.post("/api/analyze", data={"raw_text": LAB_REPORT_TEXT})
    second = client.post("/api/analyze", data={"raw_text": LAB_REPORT_TEXT})
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["code"] == "rate_limit_exceeded"


def test_rate_status_endpoint():
    response = client.get("/api/rate-status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["daily_limit"] >= payload["remaining"]


def test_health_reports_retrieval_dependency():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert "retrieval" in payload["dependencies"]
