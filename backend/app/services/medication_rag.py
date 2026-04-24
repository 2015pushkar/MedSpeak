"""Medication retrieval, grounding, and corpus-ingestion utilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.data.medication_seed import SEED_MEDICATIONS, SeedMedication
from app.schemas import MedicationEvidence
from app.tools.openfda import MedicationLabelDocument, OpenFDATool

PARENT_CHUNK_SIZE = 650
PARENT_CHUNK_TARGET = 550
CHILD_CHUNK_SIZE = 160
CHILD_CHUNK_OVERLAP = 30
CHUNKING_VERSION = "parent-child-v1"
MANIFEST_VERSION = 1


def normalize_medication_name(value: str) -> str:
    """Lowercase and normalize a medication string for matching."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def medication_name_candidates(value: str) -> list[str]:
    """Expand one medication mention into normalized lookup candidates."""
    raw = re.sub(r"\s+", " ", value).strip()
    if not raw:
        return []

    candidates = [raw]
    parenthetical = re.search(r"^(?P<outside>.+?)\s*\((?P<inside>[^)]+)\)$", raw)
    if parenthetical:
        outside = parenthetical.group("outside").strip()
        inside = parenthetical.group("inside").strip()
        if outside:
            candidates.append(outside)
        if inside:
            candidates.append(inside)

    normalized: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for variant in _candidate_variants(candidate):
            if not variant or variant in seen:
                continue
            seen.add(variant)
            normalized.append(variant)
    return normalized


def _candidate_variants(value: str) -> list[str]:
    """Generate normalized variants with and without dosage/form wording."""
    variants: list[str] = []
    normalized = normalize_medication_name(value)
    if normalized:
        variants.append(normalized)

    stripped = _strip_strength_and_form(value)
    normalized_stripped = normalize_medication_name(stripped)
    if normalized_stripped and normalized_stripped not in variants:
        variants.append(normalized_stripped)
    return variants


def _strip_strength_and_form(value: str) -> str:
    """Remove dosage strengths and dosage forms from a medication string."""
    cleaned = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|l|meq|iu|units?)\b", " ", value, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\b(?:tablet|tablets|tab|tabs|capsule|capsules|cap|caps|injection|solution|suspension|cream|ointment|patch|spray|drops?)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ,;")


def shorten_text(value: str, limit: int = 220) -> str:
    """Trim long text so evidence snippets stay readable in the API response."""
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 3].rstrip()}..."


def split_sentences(value: str) -> list[str]:
    """Split text into sentence-like chunks for chunking and summarization."""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", value) if part.strip()]
    if parts:
        return parts
    cleaned = value.strip()
    return [cleaned] if cleaned else []


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Score the angle similarity between two embedding vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def token_chunks(text: str, *, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping token windows for vector retrieval."""
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


def _hash_payload(payload: Any) -> str:
    """Create a stable content hash for manifest and rebuild checks."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_text_block(value: str) -> str:
    """Collapse whitespace before chunking or persistence."""
    return re.sub(r"\s+", " ", value).strip()


def split_parent_chunks(
    text: str,
    *,
    chunk_size: int = PARENT_CHUNK_SIZE,
    target_size: int = PARENT_CHUNK_TARGET,
) -> list[str]:
    """Build larger parent chunks that preserve more section context."""
    if not text.strip():
        return []

    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n+", text) if block.strip()]
    blocks = raw_blocks or [text.strip()]
    parents: list[str] = []

    for block in blocks:
        normalized = _normalize_text_block(block)
        if not normalized:
            continue
        if len(re.findall(r"\S+", normalized)) <= chunk_size:
            parents.append(normalized)
            continue

        sentences = split_sentences(normalized)
        if len(sentences) <= 1:
            parents.extend(token_chunks(normalized, chunk_size=chunk_size, overlap=0))
            continue

        current: list[str] = []
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = len(re.findall(r"\S+", sentence))
            if current and current_tokens + sentence_tokens > chunk_size:
                parents.append(" ".join(current).strip())
                current = [sentence]
                current_tokens = sentence_tokens
                continue

            current.append(sentence)
            current_tokens += sentence_tokens
            if current_tokens >= target_size:
                parents.append(" ".join(current).strip())
                current = []
                current_tokens = 0

        if current:
            parents.append(" ".join(current).strip())

    return [parent for parent in parents if parent]


@dataclass(frozen=True)
class MedicationChunk:
    """Stored child chunk plus the parent context it came from."""

    chunk_id: str
    parent_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    label_section: str
    text: str
    parent_text: str
    document_version: str | None = None
    source: str = "openfda"


@dataclass(frozen=True)
class RetrievedMedicationChunk:
    """Retrieved chunk enriched with score and storage metadata."""

    chunk_id: str
    parent_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    label_section: str
    text: str
    parent_text: str
    score: float
    document_version: str | None = None
    source: str = "chromadb"


@dataclass(frozen=True)
class MedicationGroundingPayload:
    """Grounded medication summary assembled from retrieved evidence."""

    purpose: str
    common_side_effects: list[str]
    cautions: list[str]
    evidence: list[MedicationEvidence]
    backend: str


@dataclass(frozen=True)
class MedicationRetrievalResult:
    """Retrieval result plus fallback details for observability."""

    chunks: list[RetrievedMedicationChunk]
    resolved_name: str | None
    backend: str
    partial_reason: str | None = None


class BaseEmbedder:
    """Minimal interface shared by all embedding backends."""

    backend = "custom"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more text inputs into vector space."""
        raise NotImplementedError

    @property
    def signature(self) -> str:
        """Return a manifest-friendly identifier for the embedder."""
        return self.backend


class DeterministicEmbedder(BaseEmbedder):
    """Offline-safe embedder used when OpenAI embeddings are unavailable."""

    backend = "deterministic"

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed text with a deterministic hashing scheme for local development."""
        return [self._embed(text) for text in texts]

    def _embed(self, text: str, dimensions: int = 96) -> list[float]:
        """Build a normalized bag-of-features vector from text tokens and n-grams."""
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
    """Production embedder backed by the OpenAI embeddings API."""

    backend = "openai"

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        """Store the shared OpenAI client and embedding model name."""
        self.client = client
        self.model = model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Delegate batch embedding generation to OpenAI."""
        if not texts:
            return []
        response = await self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    @property
    def signature(self) -> str:
        """Return a versioned identifier for manifest rebuild logic."""
        return f"{self.backend}:{self.model}"


class InMemoryMedicationVectorStore:
    """Simple in-memory vector store used for tests and local fallbacks."""

    backend = "memory"

    def __init__(self) -> None:
        """Initialize the in-memory record dictionary."""
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(self, chunks: list[MedicationChunk], embeddings: list[list[float]]) -> None:
        """Insert or replace chunk records and their embeddings."""
        for chunk, embedding in zip(chunks, embeddings):
            self._records[chunk.chunk_id] = {
                "embedding": embedding,
                "parent_id": chunk.parent_id,
                "canonical_name": chunk.canonical_name,
                "aliases": list(chunk.aliases),
                "label_section": chunk.label_section,
                "text": chunk.text,
                "parent_text": chunk.parent_text,
                "document_version": chunk.document_version,
                "source": chunk.source,
            }

    def delete(self, *, canonical_name: str | None = None, chunk_ids: list[str] | None = None) -> int:
        """Delete chunk records by medication name or explicit chunk ids."""
        removed = 0
        if chunk_ids:
            for chunk_id in chunk_ids:
                if chunk_id in self._records:
                    del self._records[chunk_id]
                    removed += 1
            return removed

        if canonical_name is None:
            return 0

        doomed = [chunk_id for chunk_id, payload in self._records.items() if payload["canonical_name"] == canonical_name]
        for chunk_id in doomed:
            del self._records[chunk_id]
        return len(doomed)

    def count(self) -> int:
        """Return how many chunk records are stored."""
        return len(self._records)

    def query(self, embedding: list[float], *, canonical_name: str | None, top_k: int) -> list[RetrievedMedicationChunk]:
        """Score stored chunks against a query embedding and return the top hits."""
        scored: list[RetrievedMedicationChunk] = []
        for chunk_id, payload in self._records.items():
            if canonical_name and payload["canonical_name"] != canonical_name:
                continue
            score = cosine_similarity(embedding, payload["embedding"])
            scored.append(
                RetrievedMedicationChunk(
                    chunk_id=chunk_id,
                    parent_id=payload.get("parent_id", chunk_id),
                    canonical_name=payload["canonical_name"],
                    aliases=tuple(payload["aliases"]),
                    label_section=payload["label_section"],
                    text=payload["text"],
                    parent_text=payload.get("parent_text", payload["text"]),
                    score=score,
                    document_version=payload.get("document_version"),
                    source=self.backend,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def status(self) -> dict[str, Any]:
        """Return a lightweight health summary for the current store."""
        return {"status": "ok", "backend": self.backend, "count": self.count()}


class FileMedicationVectorStore(InMemoryMedicationVectorStore):
    """JSON-backed vector store used when ChromaDB is unavailable."""

    backend = "json"

    def __init__(self, persist_directory: str, filename: str = "medication_labels.json") -> None:
        """Point the store at its persistence directory and JSON file."""
        super().__init__()
        self.persist_directory = Path(persist_directory)
        self.file_path = self.persist_directory / filename
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Lazy-load the JSON file so startup stays cheap."""
        if self._loaded:
            return
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        if self.file_path.exists():
            payload = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._records = payload.get("records", {})
        self._loaded = True

    def _persist(self) -> None:
        """Write the current in-memory records back to disk."""
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(json.dumps({"records": self._records}, indent=2), encoding="utf-8")

    def upsert(self, chunks: list[MedicationChunk], embeddings: list[list[float]]) -> None:
        """Update records in memory, then persist them to disk."""
        self._ensure_loaded()
        super().upsert(chunks, embeddings)
        self._persist()

    def delete(self, *, canonical_name: str | None = None, chunk_ids: list[str] | None = None) -> int:
        """Delete records and persist only when something changed."""
        self._ensure_loaded()
        removed = super().delete(canonical_name=canonical_name, chunk_ids=chunk_ids)
        if removed:
            self._persist()
        return removed

    def count(self) -> int:
        """Return the number of JSON-backed records."""
        self._ensure_loaded()
        return super().count()

    def query(self, embedding: list[float], *, canonical_name: str | None, top_k: int) -> list[RetrievedMedicationChunk]:
        """Load records if needed, then delegate to the in-memory query logic."""
        self._ensure_loaded()
        return super().query(embedding, canonical_name=canonical_name, top_k=top_k)

    def status(self) -> dict[str, Any]:
        """Return store health plus the JSON file path."""
        self._ensure_loaded()
        return {"status": "ok", "backend": self.backend, "count": self.count(), "path": str(self.file_path)}


class ChromaMedicationVectorStore:
    """Persistent ChromaDB-backed vector store for production retrieval."""

    backend = "chromadb"

    def __init__(self, persist_directory: str, collection_name: str = "medication_labels") -> None:
        """Store ChromaDB configuration without opening the collection yet."""
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self._collection = None
        self._unavailable_reason: str | None = None

    def _get_collection(self):
        """Create or reuse the Chroma collection on first access."""
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
        """Insert chunk vectors and metadata into ChromaDB."""
        collection = self._get_collection()
        if collection is None or not chunks:
            return
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "parent_id": chunk.parent_id,
                    "canonical_name": chunk.canonical_name,
                    "aliases": "|".join(chunk.aliases),
                    "label_section": chunk.label_section,
                    "parent_text": chunk.parent_text,
                    "document_version": chunk.document_version or "",
                    "source": chunk.source,
                }
                for chunk in chunks
            ],
        )

    def delete(self, *, canonical_name: str | None = None, chunk_ids: list[str] | None = None) -> int:
        """Delete Chroma records by ids or canonical medication name."""
        collection = self._get_collection()
        if collection is None:
            return 0
        if chunk_ids:
            collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        if canonical_name is None:
            return 0
        collection.delete(where={"canonical_name": canonical_name})
        return 0

    def count(self) -> int:
        """Return the number of vectors stored in Chroma."""
        collection = self._get_collection()
        if collection is None:
            return 0
        return collection.count()

    def query(self, embedding: list[float], *, canonical_name: str | None, top_k: int) -> list[RetrievedMedicationChunk]:
        """Query ChromaDB and convert raw payloads into typed retrieval records."""
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
                    parent_id=str(metadata.get("parent_id", chunk_id)),
                    canonical_name=str(metadata.get("canonical_name", "")),
                    aliases=tuple(alias for alias in str(metadata.get("aliases", "")).split("|") if alias),
                    label_section=str(metadata.get("label_section", "unknown")),
                    text=str(document),
                    parent_text=str(metadata.get("parent_text", document)),
                    score=similarity,
                    document_version=str(metadata.get("document_version", "")) or None,
                    source=self.backend,
                )
            )
        return results

    def status(self) -> dict[str, Any]:
        """Return Chroma availability plus the current collection count."""
        if self._get_collection() is None:
            return {"status": "degraded", "backend": self.backend, "error": self._unavailable_reason}
        return {"status": "ok", "backend": self.backend, "count": self.count()}


def build_label_chunks(document: MedicationLabelDocument) -> list[MedicationChunk]:
    """Convert one FDA label document into parent-child retrieval chunks."""
    chunks: list[MedicationChunk] = []
    slug = normalize_medication_name(document.canonical_name).replace(" ", "-")
    for label_section, text in document.sections.items():
        if not text.strip():
            continue

        for parent_index, parent_text in enumerate(split_parent_chunks(text)):
            parent_id = f"{slug}-{label_section}-p{parent_index}"
            child_chunks = token_chunks(parent_text, chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_CHUNK_OVERLAP) or [parent_text]
            for child_index, child_text in enumerate(child_chunks):
                chunks.append(
                    MedicationChunk(
                        chunk_id=f"{parent_id}-c{child_index}",
                        parent_id=parent_id,
                        canonical_name=document.canonical_name,
                        aliases=document.aliases,
                        label_section=label_section,
                        text=child_text,
                        parent_text=parent_text,
                        document_version=document.set_id,
                    )
                )
    return chunks


def build_seed_alias_map(seed_medications: tuple[SeedMedication, ...] = SEED_MEDICATIONS) -> dict[str, str]:
    """Build the starting alias map from the curated seed medication list."""
    mapping: dict[str, str] = {}
    for medication in seed_medications:
        canonical = medication.canonical_name
        for candidate in medication_name_candidates(canonical):
            mapping[candidate] = canonical
        for alias in medication.aliases:
            for candidate in medication_name_candidates(alias):
                mapping[candidate] = canonical
    return mapping


class MedicationRAGService:
    """Own the medication corpus, retrieval flow, and background ingestion tasks."""

    def __init__(
        self,
        *,
        store: Any,
        embedder: BaseEmbedder,
        alias_map: dict[str, str],
        persist_directory: str | None = None,
    ) -> None:
        """Store the retrieval backend pieces and manifest bookkeeping paths."""
        self.store = store
        self.embedder = embedder
        self.alias_map = alias_map
        self.persist_directory = Path(persist_directory) if persist_directory else None
        self.manifest_path = self.persist_directory / "medication_corpus_manifest.json" if self.persist_directory else None
        self._manifest: dict[str, Any] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._background_inflight: set[str] = set()

    @classmethod
    def from_settings(cls, settings: Settings, client: AsyncOpenAI | None) -> "MedicationRAGService":
        """Build the service with the best available embedder and vector store."""
        embedder: BaseEmbedder
        if client and settings.openai_enabled:
            embedder = OpenAIEmbedder(client, settings.openai_embedding_model)
        else:
            embedder = DeterministicEmbedder()
        preferred_store = ChromaMedicationVectorStore(settings.chroma_persist_directory)
        store = preferred_store if preferred_store.status()["status"] == "ok" else FileMedicationVectorStore(settings.chroma_persist_directory)
        return cls(
            store=store,
            embedder=embedder,
            alias_map=build_seed_alias_map(),
            persist_directory=settings.chroma_persist_directory,
        )

    def resolve_name(self, query: str) -> str | None:
        """Resolve a raw medication mention to the canonical corpus name."""
        for candidate in medication_name_candidates(query):
            resolved = self.alias_map.get(candidate)
            if resolved:
                return resolved
        return None

    def register_document_aliases(self, document: MedicationLabelDocument) -> None:
        """Add canonical and alias spellings from one document into the alias map."""
        for candidate in medication_name_candidates(document.canonical_name):
            self.alias_map[candidate] = document.canonical_name
        for alias in document.aliases:
            for candidate in medication_name_candidates(alias):
                self.alias_map[candidate] = document.canonical_name

    async def ingest_documents(self, documents: list[MedicationLabelDocument], *, source_mode: str = "manual") -> int:
        """Chunk, embed, and persist medication label documents incrementally."""
        manifest = self._load_manifest()
        ingested_chunks = 0
        changed = False

        for document in documents:
            self.register_document_aliases(document)
            document_hash = self._document_hash(document)
            entry = manifest["medications"].get(document.canonical_name)
            if (
                entry
                and entry.get("document_hash") == document_hash
                and entry.get("chunking_version") == CHUNKING_VERSION
                and entry.get("embedder_signature") == self.embedder.signature
            ):
                if entry.get("source_mode") != source_mode:
                    entry["source_mode"] = source_mode
                    changed = True
                continue

            chunks = build_label_chunks(document)
            self.store.delete(canonical_name=document.canonical_name)
            if not chunks:
                manifest["medications"].pop(document.canonical_name, None)
                changed = True
                continue

            embeddings = await self.embedder.embed_texts(
                [f"{chunk.canonical_name} {' '.join(chunk.aliases)} {chunk.label_section} {chunk.text}" for chunk in chunks]
            )
            self.store.upsert(chunks, embeddings)
            manifest["medications"][document.canonical_name] = {
                "aliases": list(document.aliases),
                "set_id": document.set_id,
                "document_hash": document_hash,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "chunking_version": CHUNKING_VERSION,
                "embedder_signature": self.embedder.signature,
                "source_mode": source_mode,
            }
            ingested_chunks += len(chunks)
            changed = True

        if changed:
            self._save_manifest(manifest)
        return ingested_chunks

    async def cache_openfda_document(self, medication_name: str, openfda_tool: OpenFDATool) -> bool:
        """Fetch one live OpenFDA label and add it to the local corpus."""
        try:
            document = await openfda_tool.fetch_label_document(medication_name)
        except Exception:
            return False
        if not document:
            return False
        ingested = await self.ingest_documents([document], source_mode="live")
        return ingested > 0

    def schedule_openfda_cache(self, medication_name: str, openfda_tool: OpenFDATool) -> bool:
        """Kick off background OpenFDA ingestion without blocking the request path."""
        task_key = normalize_medication_name(medication_name)
        if not task_key or task_key in self._background_inflight:
            return False

        self._background_inflight.add(task_key)

        async def runner() -> None:
            try:
                await self.cache_openfda_document(medication_name, openfda_tool)
            finally:
                self._background_inflight.discard(task_key)

        task = asyncio.create_task(runner())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return True

    async def wait_for_background_tasks(self) -> None:
        """Wait for any scheduled corpus-warming tasks to finish."""
        pending = list(self._background_tasks)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def retrieve(self, medication_name: str, *, top_k: int) -> MedicationRetrievalResult:
        """Resolve a medication name and return the best local retrieval hits."""
        resolved_name = self.resolve_name(medication_name)
        if resolved_name is None:
            return MedicationRetrievalResult(
                chunks=[],
                resolved_name=None,
                backend=self.store.backend,
            )

        if self.store.count() == 0:
            return MedicationRetrievalResult(
                chunks=[],
                resolved_name=resolved_name,
                backend=self.store.backend,
                partial_reason="Medication retrieval corpus is empty. Run the ingestion script before relying on RAG grounding.",
            )

        query_text = f"{medication_name} {resolved_name}"
        try:
            embedding = (await self.embedder.embed_texts([query_text]))[0]
            child_hits = self.store.query(embedding, canonical_name=resolved_name, top_k=max(top_k * 3, top_k))
            chunks = self._select_parent_contexts(child_hits, top_k=top_k)
        except Exception:
            return MedicationRetrievalResult(
                chunks=[],
                resolved_name=resolved_name,
                backend=self.store.backend,
                partial_reason="Medication retrieval was unavailable. MedSpeak fell back to live OpenFDA enrichment.",
            )
        return MedicationRetrievalResult(chunks=chunks, resolved_name=resolved_name, backend=self.store.backend)

    async def ground_medication(self, medication_name: str, fallback_purpose: str, *, top_k: int) -> MedicationGroundingPayload | None:
        """Turn retrieved label chunks into purpose, caution, and evidence fields."""
        retrieval = await self.retrieve(medication_name, top_k=top_k)
        if not retrieval.chunks:
            return None

        purpose = fallback_purpose
        side_effects: list[str] = []
        cautions: list[str] = []
        evidence: list[MedicationEvidence] = []

        for chunk in retrieval.chunks:
            parent_sentences = split_sentences(chunk.parent_text)
            parent_sentence = parent_sentences[0] if parent_sentences else shorten_text(chunk.parent_text)
            child_sentences = split_sentences(chunk.text)
            child_sentence = child_sentences[0] if child_sentences else shorten_text(chunk.text)

            if chunk.label_section == "indications_and_usage" and purpose == fallback_purpose:
                purpose = shorten_text(parent_sentence)
                evidence.append(
                    MedicationEvidence(
                        source=retrieval.backend,
                        label_section=chunk.label_section,
                        chunk_id=chunk.chunk_id,
                        snippet=shorten_text(child_sentence),
                    )
                )
            elif chunk.label_section == "adverse_reactions" and len(side_effects) < 2:
                side_effects.append(shorten_text(parent_sentence))
                if len(evidence) < 2:
                    evidence.append(
                        MedicationEvidence(
                            source=retrieval.backend,
                            label_section=chunk.label_section,
                            chunk_id=chunk.chunk_id,
                            snippet=shorten_text(child_sentence),
                        )
                    )
            elif chunk.label_section in {"warnings_and_cautions", "warnings", "drug_interactions"} and len(cautions) < 2:
                cautions.append(shorten_text(parent_sentence))
                if len(evidence) < 2:
                    evidence.append(
                        MedicationEvidence(
                            source=retrieval.backend,
                            label_section=chunk.label_section,
                            chunk_id=chunk.chunk_id,
                            snippet=shorten_text(child_sentence),
                        )
                    )

        return MedicationGroundingPayload(
            purpose=purpose,
            common_side_effects=self._dedupe(side_effects),
            cautions=self._dedupe(cautions),
            evidence=evidence[:2],
            backend=retrieval.backend,
        )

    async def ingest_seed_medications(self, openfda_tool: OpenFDATool) -> dict[str, Any]:
        """Populate or refresh the curated starter corpus from OpenFDA."""
        manifest = self._load_manifest()
        if self._manifest_needs_rebuild(manifest):
            self._clear_manifest_records(manifest)

        current_seed_names = {medication.canonical_name for medication in SEED_MEDICATIONS}
        previous_seed_names = set(manifest.get("seed_items", {}).keys())
        removed = sorted(previous_seed_names - current_seed_names)
        for canonical_name in removed:
            self._remove_manifest_entry(manifest, canonical_name)

        changed_seed_medications = [
            medication
            for medication in SEED_MEDICATIONS
            if manifest.get("seed_items", {}).get(medication.canonical_name) != self._seed_medication_hash(medication)
            or medication.canonical_name not in manifest["medications"]
        ]

        documents: list[MedicationLabelDocument] = []
        misses: list[str] = []
        for medication in changed_seed_medications:
            document = await openfda_tool.fetch_label_document(medication.canonical_name, aliases=medication.aliases)
            if not document:
                misses.append(medication.canonical_name)
                continue
            documents.append(document)

        chunk_count = await self.ingest_documents(documents, source_mode="seed")
        manifest = self._load_manifest()
        manifest["seed_hash"] = self._seed_hash(SEED_MEDICATIONS)
        manifest["seed_items"] = {
            medication.canonical_name: self._seed_medication_hash(medication) for medication in SEED_MEDICATIONS
        }
        self._save_manifest(manifest)
        return {
            "documents": len(documents),
            "chunks": chunk_count,
            "misses": misses,
            "removed": removed,
            "skipped": len(SEED_MEDICATIONS) - len(changed_seed_medications),
            "backend": self.store.backend,
            "embedding_backend": self.embedder.backend,
        }

    def healthcheck(self) -> dict[str, Any]:
        """Expose retrieval backend status and corpus metadata for health endpoints."""
        status = self.store.status()
        status["embedding_backend"] = self.embedder.backend
        status["chunking_strategy"] = CHUNKING_VERSION
        status["background_tasks"] = len(self._background_tasks)
        if self.manifest_path:
            status["manifest_path"] = str(self.manifest_path)
        return status

    def _select_parent_contexts(
        self,
        child_hits: list[RetrievedMedicationChunk],
        *,
        top_k: int,
    ) -> list[RetrievedMedicationChunk]:
        """Promote top child hits into unique parent contexts for final grounding."""
        seen_parent_ids: set[str] = set()
        selected: list[RetrievedMedicationChunk] = []
        for chunk in child_hits:
            if chunk.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(chunk.parent_id)
            selected.append(chunk)
            if len(selected) == top_k:
                break
        return selected

    def _dedupe(self, values: list[str]) -> list[str]:
        """Remove duplicate strings while preserving input order."""
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            normalized = value.lower()
            if not value or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(value)
        return unique

    def _default_manifest(self) -> dict[str, Any]:
        """Return the baseline manifest structure for corpus bookkeeping."""
        return {
            "version": MANIFEST_VERSION,
            "chunking_version": CHUNKING_VERSION,
            "embedder_signature": self.embedder.signature,
            "seed_hash": "",
            "seed_items": {},
            "medications": {},
        }

    def _load_manifest(self) -> dict[str, Any]:
        """Load the corpus manifest from disk once and cache it in memory."""
        if self._manifest is not None:
            return self._manifest
        if self.manifest_path is None:
            self._manifest = self._default_manifest()
            return self._manifest
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._manifest = {**self._default_manifest(), **payload}
            return self._manifest
        self._manifest = self._default_manifest()
        return self._manifest

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Persist the manifest after ingestion or rebuild changes."""
        manifest["version"] = MANIFEST_VERSION
        manifest["chunking_version"] = CHUNKING_VERSION
        manifest["embedder_signature"] = self.embedder.signature
        self._manifest = manifest
        if self.manifest_path is None:
            return
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _manifest_needs_rebuild(self, manifest: dict[str, Any]) -> bool:
        """Check whether version, chunking, or embedder changes require a reset."""
        return (
            manifest.get("version") != MANIFEST_VERSION
            or manifest.get("chunking_version") != CHUNKING_VERSION
            or manifest.get("embedder_signature") != self.embedder.signature
        )

    def _clear_manifest_records(self, manifest: dict[str, Any]) -> None:
        """Delete all stored medication records when a full rebuild is needed."""
        for canonical_name in list(manifest.get("medications", {}).keys()):
            self.store.delete(canonical_name=canonical_name)
        manifest.clear()
        manifest.update(self._default_manifest())
        self._save_manifest(manifest)

    def _remove_manifest_entry(self, manifest: dict[str, Any], canonical_name: str) -> None:
        """Remove one medication from the store and manifest tracking."""
        self.store.delete(canonical_name=canonical_name)
        manifest.get("medications", {}).pop(canonical_name, None)
        manifest.get("seed_items", {}).pop(canonical_name, None)
        self._save_manifest(manifest)

    def _document_hash(self, document: MedicationLabelDocument) -> str:
        """Hash one medication document so unchanged content can be skipped."""
        return _hash_payload(
            {
                "canonical_name": document.canonical_name,
                "aliases": list(document.aliases),
                "set_id": document.set_id,
                "sections": document.sections,
            }
        )

    def _seed_hash(self, medications: tuple[SeedMedication, ...]) -> str:
        """Hash the entire seed list to detect corpus-level changes."""
        return _hash_payload(
            [{"canonical_name": medication.canonical_name, "aliases": list(medication.aliases)} for medication in medications]
        )

    def _seed_medication_hash(self, medication: SeedMedication) -> str:
        """Hash one seed medication entry for incremental refresh checks."""
        return _hash_payload({"canonical_name": medication.canonical_name, "aliases": list(medication.aliases)})
