from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field


class MedicationEnrichment(BaseModel):
    purpose: str | None = None
    common_side_effects: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    fda_enriched: bool = False


class OpenFDATool:
    base_url = "https://api.fda.gov/drug/label.json"

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout

    async def lookup(self, medication_name: str) -> MedicationEnrichment:
        query = medication_name.strip()
        if not query:
            return MedicationEnrichment()

        params = {
            "search": f'openfda.brand_name:"{query}" OR openfda.generic_name:"{query}"',
            "limit": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
        except httpx.HTTPError:
            return MedicationEnrichment()

        if response.status_code == 404:
            return MedicationEnrichment()
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return MedicationEnrichment()
        label = results[0]

        purpose = self._first_sentence(label.get("indications_and_usage", []))
        side_effects = self._sentences(label.get("adverse_reactions", []), limit=3)
        cautions = self._sentences(
            label.get("warnings_and_cautions", []) or label.get("warnings", []) or label.get("drug_interactions", []),
            limit=3,
        )

        return MedicationEnrichment(
            purpose=purpose,
            common_side_effects=side_effects,
            cautions=cautions,
            fda_enriched=bool(purpose or side_effects or cautions),
        )

    def _first_sentence(self, values: list[str]) -> str | None:
        sentences = self._sentences(values, limit=1)
        return sentences[0] if sentences else None

    def _sentences(self, values: list[str], *, limit: int) -> list[str]:
        combined = " ".join(values)
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", combined) if part.strip()]
        unique: list[str] = []
        for part in parts:
            normalized = re.sub(r"\s+", " ", part)
            if normalized not in unique:
                unique.append(normalized)
            if len(unique) == limit:
                break
        return unique

