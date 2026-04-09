# MedSpeak

MedSpeak is an AI-assisted medical document explainer built as a demo-friendly monorepo with a FastAPI backend and a Next.js frontend.

It accepts PDF uploads or pasted report text, extracts structured findings, enriches medications with OpenFDA label data, and returns plain-language summaries with questions to bring to a clinician.

## Repo layout

```text
backend/   FastAPI API, analysis pipeline, tests
frontend/  Next.js app, UI tests, e2e tests
```

## Local setup

### Backend

```bash
cd backend
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

## Environment

- Backend API: `http://localhost:8000`
- Frontend app: `http://localhost:3000`

## Notes

- This MVP is a portfolio/demo build, not a compliance-ready healthcare product.
- Documents are processed ephemerally. No document history or account system is included.
- PDF extraction uses PyMuPDF. Image OCR is intentionally deferred.

