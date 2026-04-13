from __future__ import annotations

import math
import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.data.medication_seed import SEED_MEDICATIONS, SeedMedication
from app.schemas import MedicationEvidence
from app.tools.openfda import MedicationLabelDocument, OpenFDATool


def normalize_medication_name(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def shorten_text(value: str, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def split_sentences(value: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]
    if parts:
        return parts
    cleaned = value.strip()
    return [cleaned] if cleaned else []


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def token_chunks(text: str, *, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    if overlap >= chunk_size:
        overlap = max(chunk_size // 10, 1)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
        start += chunk_size - overlap
    return chunks


@dataclass(frozen=True)
class MedicationChunk:
    chunk_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    label_section: str
    text: str
    source: str = "openfda"


@dataclass(frozen=True)
class RetrievedMedicationChunk:
    chunk_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    label_section: str
    text: str
    score: float
    source: str = "chromadb"


@dataclass(frozen=True)
class MedicationGroundingPayload:
    purpose: str
    common_side_effects: list[str]
    cautions: list[str]
    evidence: list[MedicationEvidence]
    backend: str


@dataclass(frozen=True)
class MedicationRetrievalResult:
    chunks: list[RetrievedMedicationChunk]
    resolved_name: str | None
    backend: str
    partial_reason: str | None = None


class BaseEmbedder:
    backend = "custom"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class DeterministicEmbedder(BaseEmbedder):
    backend = "deterministic"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str, dimensions: int = 96) -> list[float]:
        vector = [0.0] * dimensions
        normalized = normalize_medication_name(text)
        if not normalized:
            return vector
        tokens = normalized.split()
        features = list(tokens)
        compact = normalized.replace(" ", "")
        features.extend(compact[index : index + 3] for index in range(max(len(compact) - 2, 0)))
        for feature in features:
            position = hash(feature) % dimensions
            vector[position] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbedder(BaseEmbedder):
    backend = "openai"

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]


class InMemoryMedicationVectorStore:
    backend = "memory"

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(self, chunks: list[MedicationChunk], embeddings: list[list[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self._records[chunk.chunk_id] = {
                "embedding": embedding,
                "canonical_name": chunk.canonical_name,
                "aliases": list(chunk.aliases),
                "label_section": chunk.label_section,
                "text": chunk.text,
                "source": chunk.source,
            }

    def count(self) -> int:
        return len(self._records)

    def query(self, embedding: list[float], *, canonical_name: str | None, top_k: int) -> list[RetrievedMedicationChunk]:
        scored: list[RetrievedMedicationChunk] = []
        for chunk_id, payload in self._records.items():
            if canonical_name and payload["canonical_name"] != canonical_name:
                continue
            score = cosine_similarity(embedding, payload["embedding"])
            scored.append(
                RetrievedMedicationChunk(
                    chunk_id=chunk_id,
                    canonical_name=payload["canonical_name"],
                    aliases=tuple(payload["aliases"]),
                    label_section=payload["label_section"],
                    text=payload["text"],
                    score=score,
                    source=self.backend,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def status(self) -> dict[str, Any]:
        return {"status": "ok", "backend": self.backend, "count": self.count()}


class FileMedicationVectorStore(InMemoryMedicationVectorStore):
    backend = "json"

    def __init__(self, persist_directory: str, filename: str = "medication_labels.json") -> None:
        super().__init__()
        self.persist_directory = Path(persist_directory)
        self.file_path = self.persist_directory / filename
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._records = payload.get("records", {})
        self._loaded = True

    def _persist(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps({"records": self._records}, indent=2), encoding="utf-8")

    def upsert(self, chunks: list[MedicationChunk], embeddings: list[list[float]]) -> None:
        self._ensure_loaded()
        super().upsert(chunks, embeddings)
        self._persist()

    def count(self) -> int:
        self._ensure_loaded()
        return super().count()

    def query(self, embedding: list[float], *, canonical_name: str | None, top_k: int) -> list[RetrievedMedicationChunk]:
        self._ensure_loaded()
        return super().query(embedding, canonical_name=canonical_name, top_k=top_k)

    def status(self) -> dict[str, Any]:
        self._ensure_loaded()
        return {"status": "ok", "backend": self.backend, "count": self.count(), "path": str(self.file_path)}


class ChromaMedicationVectorStore:
    backend = "chromadb"

    def __init__(self, persist_directory: str, collection_name: str = "medication_labels") -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._collection = None
        self._unavailable_reason: str | None = None

    def _get_collection(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError:
            self._unavailable_reason = "chromadb is not installed."
            return None

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def upsert(self, chunks: list[MedicationChunk], embeddings: list[list[float]]) -> None:
        collection = self._get_collection()
        if collection is None or not chunks:
            return
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "canonical_name": chunk.canonical_name,
                    "aliases": "|".join(chunk.aliases),
                    "label_section": chunk.label_section,
                    "source": chunk.source,
                }
                for chunk in chunks
            ],
        )

    def count(self) -> int:
        collection = self._get_collection()
        if collection is None:
            return 0
        return collection.count()

    def query(self, embedding: list[float], *, canonical_name: str | None, top_k: int) -> list[RetrievedMedicationChunk]:
        collection = self._get_collection()
        if collection is None or collection.count() == 0:
            return []
        where = {"canonical_name": canonical_name} if canonical_name else None
        payload = collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )
        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]
        results: list[RetrievedMedicationChunk] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            similarity = 1 - float(distance)
            results.append(
                RetrievedMedicationChunk(
                    chunk_id=chunk_id,
                    canonical_name=str(metadata.get("canonical_name", "")),
                    aliases=tuple(alias for alias in str(metadata.get("aliases", "")).split("|") if alias),
                    label_section=str(metadata.get("label_section", "unknown")),
                    text=str(document),
                    score=similarity,
                    source=self.backend,
                )
            )
        return results

    def status(self) -> dict[str, Any]:
        if self._get_collection() is None:
            return {"status": "degraded", "backend": self.backend, "error": self._unavailable_reason}
        return {"status": "ok", "backend": self.backend, "count": self.count()}


def build_label_chunks(document: MedicationLabelDocument, *, chunk_size: int = 500, overlap: int = 50) -> list[MedicationChunk]:
    chunks: list[MedicationChunk] = []
    for label_section, text in document.sections.items():
        if not text.strip():
            continue
        for index, chunk_text in enumerate(token_chunks(text, chunk_size=chunk_size, overlap=overlap)):
            chunks.append(
                MedicationChunk(
                    chunk_id=f"{normalize_medication_name(document.canonical_name).replace(' ', '-')}-{label_section}-{index}",
                    canonical_name=document.canonical_name,
                    aliases=document.aliases,
                    label_section=label_section,
                    text=chunk_text,
                )
            )
    return chunks


def build_seed_alias_map(seed_medications: tuple[SeedMedication, ...] = SEED_MEDICATIONS) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for medication in seed_medications:
        canonical = medication.canonical_name
        mapping[normalize_medication_name(canonical)] = canonical
        for alias in medication.aliases:
            mapping[normalize_medication_name(alias)] = canonical
    return mapping


class MedicationRAGService:
    def __init__(self, *, store: Any, embedder: BaseEmbedder, alias_map: dict[str, str]) -> None:
        self.store = store
        self.embedder = embedder
        self.alias_map = alias_map

    @classmethod
    def from_settings(cls, settings: Settings, client: AsyncOpenAI | None) -> "MedicationRAGService":
        embedder: BaseEmbedder
        if client and settings.openai_enabled:
            embedder = OpenAIEmbedder(client, settings.openai_embedding_model)
        else:
            embedder = DeterministicEmbedder()
        preferred_store = ChromaMedicationVectorStore(settings.chroma_persist_directory)
        store = preferred_store if preferred_store.status()["status"] == "ok" else FileMedicationVectorStore(settings.chroma_persist_directory)
        return cls(store=store, embedder=embedder, alias_map=build_seed_alias_map())

    def resolve_name(self, query: str) -> str | None:
        return self.alias_map.get(normalize_medication_name(query))

    async def ingest_documents(self, documents: list[MedicationLabelDocument]) -> int:
        chunks: list[MedicationChunk] = []
        for document in documents:
            chunks.extend(build_label_chunks(document))
        if not chunks:
            return 0
        embeddings = await self.embedder.embed_texts(
            [f"{chunk.canonical_name} {' '.join(chunk.aliases)} {chunk.label_section} {chunk.text}" for chunk in chunks]
        )
        self.store.upsert(chunks, embeddings)
        return len(chunks)

    async def retrieve(self, medication_name: str, *, top_k: int) -> MedicationRetrievalResult:
        if self.store.count() == 0:
            return MedicationRetrievalResult(
                chunks=[],
                resolved_name=self.resolve_name(medication_name),
                backend=self.store.backend,
                partial_reason="Medication retrieval corpus is empty. Run the ingestion script before relying on RAG grounding.",
            )

        resolved_name = self.resolve_name(medication_name)
        query_text = medication_name if not resolved_name else f"{medication_name} {resolved_name}"
        try:
            embedding = (await self.embedder.embed_texts([query_text]))[0]
            chunks = self.store.query(embedding, canonical_name=resolved_name, top_k=top_k)
            if not chunks and resolved_name is None:
                expanded = self.store.query(embedding, canonical_name=None, top_k=top_k * 2)
                chunks = [chunk for chunk in expanded if self._is_plausible_match(medication_name, chunk)][:top_k]
        except Exception:
            return MedicationRetrievalResult(
                chunks=[],
                resolved_name=resolved_name,
                backend=self.store.backend,
                partial_reason="Medication retrieval was unavailable. MedSpeak fell back to live OpenFDA enrichment.",
            )
        return MedicationRetrievalResult(chunks=chunks, resolved_name=resolved_name, backend=self.store.backend)

    async def ground_medication(self, medication_name: str, fallback_purpose: str, *, top_k: int) -> MedicationGroundingPayload | None:
        retrieval = await self.retrieve(medication_name, top_k=top_k)
        if not retrieval.chunks:
            return None

        purpose = fallback_purpose
        side_effects: list[str] = []
        cautions: list[str] = []
        evidence: list[MedicationEvidence] = []

        for chunk in retrieval.chunks:
            sentence = split_sentences(chunk.text)[0] if split_sentences(chunk.text) else shorten_text(chunk.text)
            evidence.append(
                MedicationEvidence(
                    source=retrieval.backend,
                    label_section=chunk.label_section,
                    chunk_id=chunk.chunk_id,
                    snippet=shorten_text(sentence),
                )
            )
            if chunk.label_section == "indications_and_usage" and purpose == fallback_purpose:
                purpose = shorten_text(sentence)
            elif chunk.label_section == "adverse_reactions" and len(side_effects) < 2:
                side_effects.append(shorten_text(sentence))
            elif chunk.label_section in {"warnings_and_cautions", "warnings", "drug_interactions", "dosage_and_administration"} and len(cautions) < 2:
                cautions.append(shorten_text(sentence))

        return MedicationGroundingPayload(
            purpose=purpose,
            common_side_effects=self._dedupe(side_effects),
            cautions=self._dedupe(cautions),
            evidence=evidence[:2],
            backend=retrieval.backend,
        )

    async def ingest_seed_medications(self, openfda_tool: OpenFDATool) -> dict[str, Any]:
        documents: list[MedicationLabelDocument] = []
        misses: list[str] = []
        for medication in SEED_MEDICATIONS:
            document = await openfda_tool.fetch_label_document(medication.canonical_name, aliases=medication.aliases)
            if not document:
                misses.append(medication.canonical_name)
                continue
            documents.append(document)
        chunk_count = await self.ingest_documents(documents)
        return {
            "documents": len(documents),
            "chunks": chunk_count,
            "misses": misses,
            "backend": self.store.backend,
            "embedding_backend": self.embedder.backend,
        }

    def healthcheck(self) -> dict[str, Any]:
        status = self.store.status()
        status["embedding_backend"] = self.embedder.backend
        return status

    def _is_plausible_match(self, medication_name: str, chunk: RetrievedMedicationChunk) -> bool:
        normalized_query = normalize_medication_name(medication_name)
        candidates = {normalize_medication_name(chunk.canonical_name), *(normalize_medication_name(alias) for alias in chunk.aliases)}
        if normalized_query in candidates:
            return True
        query_tokens = set(normalized_query.split())
        return any(query_tokens and query_tokens.issubset(set(candidate.split())) for candidate in candidates)

    def _dedupe(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            normalized = value.lower()
            if not value or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value)
        return unique
