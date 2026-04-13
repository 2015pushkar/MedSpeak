# MedSpeak

MedSpeak is an AI-assisted medical document explainer built as a FastAPI + Next.js monorepo. It accepts PDF uploads or pasted report text, extracts structured findings, and returns plain-language explanations with clinician-oriented follow-up questions.

## Implemented Now

### Backend
- FastAPI API with `GET /api/health`, `GET /api/rate-status`, and `POST /api/analyze`
- LangGraph orchestration with classifier, lab, medication, diagnosis, and synthesis nodes
- Server-side PDF extraction with PyMuPDF
- OpenFDA live medication enrichment
- Retrieval-backed medication grounding with evidence snippets
- Bounded OpenAI retries and deterministic fallback behavior
- Safety review that catches dosage / diagnosis / treatment-like language and rewrites or replaces unsafe drafts
- Upstash Redis rate limiting with in-memory fallback

### Retrieval
- Seed corpus of 10 medications ingested from OpenFDA
- Section-aware chunking at `500` tokens with `50` overlap
- Persistent local vector store under `backend/data/chroma`
- Preferred ChromaDB backend with a persistent JSON fallback when `chromadb` cannot be compiled locally
- Retrieval evaluation script with measured top-1 / top-3 hit rate output

### Frontend
- Next.js App Router UI for pasted text and PDF uploads
- Plain-language result cards for labs, medications, diagnoses, warnings, and doctor questions
- Grounding badges and evidence snippets on medication cards

## Planned Next
- Verify the native ChromaDB backend on a machine with the required C++ build tools
- Re-run ingestion and retrieval evaluation with OpenAI embeddings enabled
- Expand the eval set beyond the 10-query interview sample
- Add human review for prompt wording and healthcare-safety tone

## Repo Layout

```text
backend/   FastAPI API, retrieval pipeline, scripts, tests
frontend/  Next.js app, UI tests, e2e tests
PLANS.md   detailed review, measured retrieval metric, next-phase checkpoints
```

## Local Setup

### Backend

```bash
cd backend
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

### Retrieval Scripts

```bash
cd backend
.venv\Scripts\python.exe scripts\ingest_openfda_labels.py
.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

## Docker

Backend container build:

```bash
cd backend
docker build -t medspeak-backend .
```

## Environment

- Backend API: `http://localhost:8000`
- Frontend app: `http://localhost:3000`

## Notes

- This is an interview/demo build, not a compliance-ready healthcare product.
- Documents are processed ephemerally. No accounts or history are included.
- PDF extraction remains on the backend. OCR is intentionally out of scope.
- On this machine, OpenFDA ingestion and retrieval evaluation ran successfully using the persistent JSON vector-store fallback because `chromadb` could not compile without local MSVC build tools.
