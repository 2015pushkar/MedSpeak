from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field


class MedicationEnrichment(BaseModel):
    purpose: str | None = None
    common_side_effects: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    fda_enriched: bool = False


class MedicationLabelDocument(BaseModel):
    canonical_name: str
    aliases: tuple[str, ...] = ()
    set_id: str | None = None
    sections: dict[str, str] = Field(default_factory=dict)


class OpenFDATool:
    base_url = "https://api.fda.gov/drug/label.json"

    def __init__(self, timeout: float = 6.0) -> None:
        self.timeout = timeout

    async def lookup(self, medication_name: str) -> MedicationEnrichment:
        label = await self._fetch_label_payload(medication_name)
        if not label:
            return MedicationEnrichment()

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

    async def fetch_label_document(
        self,
        medication_name: str,
        *,
        aliases: tuple[str, ...] = (),
    ) -> MedicationLabelDocument | None:
        label = await self._fetch_label_payload(medication_name)
        if not label:
            return None

        openfda = label.get("openfda", {})
        discovered_aliases = set(aliases)
        for field_name in ("brand_name", "generic_name", "substance_name"):
            for value in openfda.get(field_name, []) or []:
                cleaned = re.sub(r"\s+", " ", value).strip()
                if cleaned:
                    discovered_aliases.add(cleaned)

        sections = {
            "indications_and_usage": self._combine(label.get("indications_and_usage", [])),
            "dosage_and_administration": self._combine(label.get("dosage_and_administration", [])),
            "warnings_and_cautions": self._combine(label.get("warnings_and_cautions", []) or label.get("warnings", [])),
            "drug_interactions": self._combine(label.get("drug_interactions", [])),
            "adverse_reactions": self._combine(label.get("adverse_reactions", [])),
        }
        cleaned_sections = {name: value for name, value in sections.items() if value}
        if not cleaned_sections:
            return None

        return MedicationLabelDocument(
            canonical_name=medication_name.strip().title(),
            aliases=tuple(sorted(discovered_aliases)),
            set_id=label.get("set_id"),
            sections=cleaned_sections,
        )

    async def _fetch_label_payload(self, medication_name: str) -> dict | None:
        query = medication_name.strip()
        if not query:
            return None

        params = {
            "search": f'openfda.brand_name:"{query}" OR openfda.generic_name:"{query}"',
            "limit": 1,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(self.base_url, params=params)
        except httpx.HTTPError:
            return None

        if response.status_code == 404:
            return None
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results") or []
        if not results:
            return None
        return results[0]

    def _first_sentence(self, values: list[str]) -> str | None:
        sentences = self._sentences(values, limit=1)
        return sentences[0] if sentences else None

    def _combine(self, values: list[str]) -> str:
        return re.sub(r"\s+", " ", " ".join(values)).strip()

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
