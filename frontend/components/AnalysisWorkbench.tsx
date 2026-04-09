"use client";

import { FormEvent, startTransition, useEffect, useState } from "react";

import { analyzeDocument, getRateStatus } from "@/lib/api";
import type { AnalysisResponse, ApiErrorShape, RateStatus } from "@/lib/types";
import { ResultPanel } from "@/components/ResultPanel";

type InputMode = "text" | "file";

function isApiErrorShape(error: unknown): error is ApiErrorShape {
  return typeof error === "object" && error !== null && "message" in error && "code" in error;
}

function formatReset(resetAt: string | undefined) {
  if (!resetAt) {
    return "later today";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(new Date(resetAt));
}

export function AnalysisWorkbench() {
  const [mode, setMode] = useState<InputMode>("text");
  const [rawText, setRawText] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [rateStatus, setRateStatus] = useState<RateStatus | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [fieldMessage, setFieldMessage] = useState("");

  useEffect(() => {
    void loadRateStatus();
  }, []);

  async function loadRateStatus() {
    try {
      const nextRateStatus = await getRateStatus();
      setRateStatus(nextRateStatus);
    } catch {
      setRateStatus(null);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setErrorMessage("");
    setFieldMessage("");
    setResult(null);

    if (rateStatus && rateStatus.remaining === 0) {
      setErrorMessage(`The free daily limit is already used. Try again after ${formatReset(rateStatus.reset_at)}.`);
      return;
    }

    const formData = new FormData();
    if (mode === "text") {
      if (!rawText.trim()) {
        setFieldMessage("Paste report text before starting an analysis.");
        return;
      }
      formData.append("raw_text", rawText.trim());
    } else {
      if (!selectedFile) {
        setFieldMessage("Choose a PDF before starting an analysis.");
        return;
      }
      formData.append("file", selectedFile);
    }

    setLoading(true);
    try {
      const nextResult = await analyzeDocument(formData);
      startTransition(() => {
        setResult(nextResult);
        setRateStatus({
          remaining: nextResult.meta.rate_limit_remaining,
          daily_limit: nextResult.meta.daily_limit,
          reset_at: nextResult.meta.rate_limit_reset_at,
        });
      });
    } catch (error) {
      if (isApiErrorShape(error)) {
        setErrorMessage(error.message);
        if (error.code === "rate_limit_exceeded") {
          const resetAt = typeof error.details?.reset_at === "string" ? error.details.reset_at : undefined;
          setRateStatus((current) => ({
            remaining: 0,
            daily_limit: typeof error.details?.daily_limit === "number" ? error.details.daily_limit : current?.daily_limit ?? 5,
            reset_at: resetAt ?? current?.reset_at ?? new Date().toISOString(),
          }));
        }
      } else {
        setErrorMessage("The analysis request failed before MedSpeak could return a structured response.");
      }
    } finally {
      setLoading(false);
    }
  }

  const remainingLabel = rateStatus ? `${rateStatus.remaining} of ${rateStatus.daily_limit} free analyses left today` : "Checking today's free limit";
  const quotaLocked = Boolean(rateStatus && rateStatus.remaining === 0);

  return (
    <div className="space-y-8">
      <section className="grid gap-6 lg:grid-cols-[1.1fr_0.9fr]">
        <form onSubmit={handleSubmit} className="rounded-[2rem] border border-white/60 bg-white/88 p-6 shadow-panel backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.35em] text-ink/45">Upload Or Paste</p>
              <h2 className="mt-3 font-serif text-3xl text-ink">Turn the medical wording into plain language</h2>
            </div>
            <div className="rounded-full bg-ink px-4 py-2 text-xs uppercase tracking-[0.28em] text-mist">{remainingLabel}</div>
          </div>

          <div className="mt-6 inline-flex rounded-full border border-ink/10 bg-mist p-1">
            <button
              type="button"
              onClick={() => {
                setMode("text");
                setFieldMessage("");
              }}
              className={`rounded-full px-4 py-2 text-sm font-semibold ${mode === "text" ? "bg-ink text-mist" : "text-ink/70"}`}
            >
              Paste Text
            </button>
            <button
              type="button"
              onClick={() => {
                setMode("file");
                setFieldMessage("");
              }}
              className={`rounded-full px-4 py-2 text-sm font-semibold ${mode === "file" ? "bg-ink text-mist" : "text-ink/70"}`}
            >
              Upload PDF
            </button>
          </div>

          {mode === "text" ? (
            <label className="mt-6 block">
              <span className="text-sm font-semibold text-ink">Paste report text</span>
              <textarea
                value={rawText}
                onChange={(event) => setRawText(event.target.value)}
                rows={12}
                placeholder="Glucose: 142 mg/dL (70-100)..."
                className="mt-3 w-full rounded-[1.6rem] border border-ink/10 bg-mist/70 px-4 py-4 text-sm text-ink outline-none transition focus:border-ink/30"
              />
            </label>
          ) : (
            <label className="mt-6 flex min-h-60 cursor-pointer flex-col items-center justify-center rounded-[1.8rem] border border-dashed border-ink/20 bg-mist/70 p-6 text-center">
              <span className="text-sm font-semibold text-ink">Upload PDF report</span>
              <span className="mt-2 max-w-xs text-sm leading-6 text-ink/65">
                MedSpeak supports text-based PDFs in this MVP. Scanned image PDFs are intentionally deferred.
              </span>
              <input
                type="file"
                accept="application/pdf,.pdf"
                onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                className="mt-5 block text-sm text-ink"
              />
              {selectedFile && <span className="mt-4 rounded-full bg-white px-4 py-2 text-xs uppercase tracking-[0.2em] text-ink/75">{selectedFile.name}</span>}
            </label>
          )}

          {fieldMessage && <p className="mt-4 text-sm text-coral">{fieldMessage}</p>}
          {errorMessage && <p className="mt-4 rounded-2xl bg-coral/12 px-4 py-3 text-sm text-ink">{errorMessage}</p>}

          <div className="mt-6 flex flex-wrap items-center gap-4">
            <button
              type="submit"
              disabled={loading || quotaLocked}
              className="rounded-full bg-ink px-6 py-3 text-sm font-semibold uppercase tracking-[0.24em] text-mist transition disabled:cursor-not-allowed disabled:bg-ink/35"
            >
              {loading ? "Analyzing..." : "Explain This Report"}
            </button>
            <p className="text-sm leading-6 text-ink/65">
              Educational use only. Always review medical decisions with a licensed clinician.
            </p>
          </div>
        </form>

        <aside className="rounded-[2rem] border border-white/50 bg-gradient-to-br from-ink via-ink to-leaf p-6 text-mist shadow-panel">
          <p className="text-xs uppercase tracking-[0.35em] text-mist/55">How It Works</p>
          <div className="mt-5 space-y-4 text-sm leading-7 text-mist/90">
            <p>1. Intake accepts pasted text or a PDF and validates the daily free quota.</p>
            <p>2. The backend extracts text, classifies the report, and routes it through lab, medication, and diagnosis specialists.</p>
            <p>3. MedSpeak returns plain-language cards, OpenFDA-backed cautions, and questions to bring to your next appointment.</p>
          </div>
          <div className="mt-6 rounded-3xl border border-white/10 bg-white/5 p-4">
            <p className="font-semibold">Current MVP boundaries</p>
            <p className="mt-2 text-sm leading-6 text-mist/80">No accounts, no history, no export, and no image OCR. Documents are processed ephemerally for demo use.</p>
          </div>
        </aside>
      </section>

      {loading && (
        <section className="rounded-[2rem] border border-white/60 bg-white/85 p-6 text-ink shadow-panel">
          <p className="text-xs uppercase tracking-[0.35em] text-ink/45">In Progress</p>
          <h2 className="mt-3 font-serif text-3xl">Translating the report into everyday language</h2>
          <p className="mt-3 max-w-2xl text-sm leading-7 text-ink/70">
            MedSpeak is extracting the document, checking for lab patterns, medication names, and diagnosis terms, then drafting questions you can bring to a clinician.
          </p>
        </section>
      )}

      {result && !loading && <ResultPanel result={result} />}
    </div>
  );
}
