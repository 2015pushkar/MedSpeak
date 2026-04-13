# MedSpeak Detailed Review And Roadmap

## Current Implemented State

### Backend
- FastAPI endpoints for `GET /api/health`, `GET /api/rate-status`, and `POST /api/analyze`.
- LangGraph pipeline with classifier, lab, medication, diagnosis, and synthesis nodes.
- PDF text extraction on the backend with PyMuPDF.
- Per-IP daily rate limiting with Upstash Redis when configured and in-memory fallback otherwise.
- Bounded OpenAI retry behavior with deterministic fallback when LLM calls fail.
- Deterministic post-synthesis safety review that flags diagnosis, treatment, and dosage language, then rewrites or replaces unsafe drafts.
- Medication grounding pipeline with three states:
  - `rag`: retrieved from the local medication corpus.
  - `openfda_live`: grounded from a live OpenFDA label lookup.
  - `text_only`: document-only fallback when no grounded source is available.
- Medication evidence objects returned in the API with `source`, `label_section`, `chunk_id`, and `snippet`.

### Retrieval / RAG
- Offline ingestion script at [backend/scripts/ingest_openfda_labels.py](/c:/Users/Pushk/OneDrive/Documents/GitHub/MedSpeak/backend/scripts/ingest_openfda_labels.py).
- Seed corpus of 10 medications ingested from OpenFDA and persisted under [backend/data/chroma/medication_labels.json](/c:/Users/Pushk/OneDrive/Documents/GitHub/MedSpeak/backend/data/chroma/medication_labels.json).
- Section-aware chunking at roughly `500` tokens with `50` overlap.
- Retrieval service that prefers ChromaDB when available and falls back to a persistent JSON vector store on machines where `chromadb` cannot compile.
- Deterministic embeddings for offline/dev mode; OpenAI embeddings are used automatically when an API key is configured.

### Frontend
- Next.js UI for pasted text and PDF uploads.
- Result cards for labs, medications, diagnoses, questions for the doctor, disclaimer, and partial-data notes.
- Medication cards now surface grounding source and evidence snippets directly in the UI.

### Verification
- Backend tests: `19` passing.
- Frontend Vitest: `5` passing.
- Playwright e2e: `2` passing.

## Target Interview-Ready State
- Keep the current MVP boundaries: no auth, no history, no OCR, no compliance buildout.
- Use MedSpeak as an interview-grade example of:
  - multi-agent orchestration,
  - explicit RAG ingestion/chunking/embedding/retrieval,
  - grounded medication evidence,
  - graceful fallback behavior,
  - healthcare-oriented safety controls.
- Keep the API stable around `POST /api/analyze` while improving grounding quality and demo quality.

## Gap Matrix
| Area | Status | Notes |
| --- | --- | --- |
| Multi-agent orchestration | Implemented | LangGraph graph is live in the backend. |
| Medical safety layer | Implemented | Deterministic review plus rewrite/canned fallback added. |
| OpenFDA live enrichment | Implemented | Still used as the first fallback when retrieval misses. |
| Persistent local retrieval corpus | Implemented | Persisted as JSON fallback in `backend/data/chroma`. |
| ChromaDB native backend | Partially implemented | Code path exists and is preferred, but `chromadb` installation failed on this machine because `chroma-hnswlib` requires MSVC build tools. |
| Retrieval evaluation story | Implemented | Script exists and a local sample measurement was recorded. |
| UI grounding transparency | Implemented | Medication cards now show grounding source and snippets. |
| Docker packaging | Implemented | Backend Dockerfile added. |
| Clinician-reviewed medical wording | Human gate | Prompt/safety wording still needs domain review. |
| Broader medication coverage | Next phase | Current corpus is intentionally limited to 10 medications. |

## Phase Plan

### Phase 1: Interview-Ready MVP
- Completed:
  - retrieval service,
  - ingestion/eval scripts,
  - safety review,
  - grounding-aware API and UI,
  - tests and Dockerfile.

### Phase 2: Retrieval Hardening
- Install MSVC build tools or use a prebuilt environment so the ChromaDB backend can be exercised directly instead of the JSON fallback.
- Re-run ingestion with OpenAI embeddings enabled and compare retrieval quality against deterministic offline mode.
- Expand the eval set from brand/generic aliases to harder phrasing and mixed-document mentions.

### Phase 3: Medical Quality Review
- Review disclaimer wording, refusal patterns, and doctor-question phrasing with a medically informed reviewer.
- Review a small set of golden documents manually and capture accepted examples for demo use.

## Retrieval Evaluation Story
- Seed corpus: 10 medications from OpenFDA.
- Eval set: 10 labeled medication queries covering canonical names and common aliases.
- Hit definition:
  - `top_1_hit_rate`: expected medication is the first retrieved medication.
  - `top_3_hit_rate`: expected medication appears anywhere in the top 3 retrieved results.
- Recorded local result after ingestion on `2026-04-13`:
  - `top_1_hit_rate = 1.0`
  - `top_3_hit_rate = 1.0`
  - `total_eval_cases = 10`
  - miss buckets: `alias_mismatch=0`, `corpus_coverage_gap=0`, `weak_chunk_match=0`
- Interview-safe wording:
  - "I sampled retrieval over a labeled 10-query medication set built from the seeded corpus and saw 100% top-3 hit rate in the current configuration."
- Important caveat:
  - That measurement was produced locally with the deterministic embedding fallback because OpenAI embeddings were not configured during the run.

## Human-in-the-Loop Checkpoints
- Confirm the 10-medication seed list is the right interview/demo corpus.
- Review disclaimer wording and the exact refusal tone for diagnosis, treatment, and dosing language.
- Review the 10 retrieval eval cases and confirm the expected medication labels.
- Review a small set of real outputs after ingestion and after safety enforcement to ensure the tone is useful and not alarmist.

## Acceptance Checklist
- [x] Grounded medication evidence is present in backend responses.
- [x] Medication UI shows grounding state and evidence snippets.
- [x] Retrieval ingestion script exists and runs.
- [x] Retrieval eval script exists and reports top-1/top-3 hit rate.
- [x] Safety layer can rewrite or replace unsafe drafts.
- [x] Backend, frontend, and e2e test suites pass.
- [ ] ChromaDB native backend verified on a machine with required build tooling.
- [ ] Prompt and safety wording reviewed by a human with healthcare context.
