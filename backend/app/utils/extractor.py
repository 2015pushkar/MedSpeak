from __future__ import annotations

import fitz
from pydantic import BaseModel

from app.errors import AppError


class ExtractedDocument(BaseModel):
    text: str
    page_count: int = 0
    truncated: bool = False


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def validate_raw_text(raw_text: str | None, max_chars: int) -> ExtractedDocument:
    if raw_text is None:
        raise AppError(status_code=400, code="missing_input", message="Provide either a PDF file or raw_text.")
    cleaned = normalize_text(raw_text)
    if not cleaned:
        raise AppError(status_code=400, code="empty_text", message="raw_text was provided but did not contain usable text.")
    if len(cleaned) > max_chars:
        raise AppError(
            status_code=413,
            code="text_too_large",
            message="The pasted text exceeds the maximum supported size for this demo.",
            details={"max_characters": max_chars},
        )
    return ExtractedDocument(text=cleaned)


def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_chars: int) -> ExtractedDocument:
    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except RuntimeError as exc:
        raise AppError(
            status_code=400,
            code="invalid_pdf",
            message="The uploaded file could not be parsed as a PDF.",
        ) from exc

    page_text = [page.get_text("text") for page in document]
    cleaned = normalize_text("\n".join(page_text))
    if not cleaned:
        raise AppError(
            status_code=422,
            code="non_extractable_pdf",
            message="This PDF does not contain extractable text. Image OCR is not supported in this MVP.",
        )

    truncated = len(cleaned) > max_chars
    if truncated:
        cleaned = cleaned[:max_chars].strip()

    return ExtractedDocument(text=cleaned, page_count=len(document), truncated=truncated)

