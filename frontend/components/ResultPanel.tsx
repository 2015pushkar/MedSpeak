import type { ReactNode } from "react";

import type { AnalysisResponse, GroundingStatus, JudgeStatus, LabStatus, MedicationStatus } from "@/lib/types";

const statusTone: Record<LabStatus, string> = {
  low: "bg-coral/15 text-coral",
  normal: "bg-leaf/12 text-leaf",
  high: "bg-gold/20 text-ink",
  unknown: "bg-ink/8 text-ink/60",
};

const groundingTone: Record<GroundingStatus, string> = {
  rag: "bg-leaf/15 text-leaf",
  openfda_live: "bg-gold/20 text-ink",
  text_only: "bg-ink/8 text-ink/60",
};

const groundingLabel: Record<GroundingStatus, string> = {
  rag: "RAG grounded",
  openfda_live: "OpenFDA live",
  text_only: "Text only",
};

const medicationStatusTone: Record<MedicationStatus, string> = {
  current: "bg-leaf/15 text-leaf",
  historical: "bg-ink/8 text-ink/70",
  otc_prn: "bg-gold/18 text-ink",
  unclear: "bg-coral/12 text-coral",
};

const medicationStatusLabel: Record<MedicationStatus, string> = {
  current: "Current",
  historical: "Historical",
  otc_prn: "OTC / PRN",
  unclear: "Unclear",
};

const judgeLabel: Record<JudgeStatus, string> = {
  passed: "Passed",
  failed: "Needs review",
  unavailable: "Unavailable",
  skipped: "Skipped",
};

function ChevronDownIcon({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className={className}>
      <path d="M5 7.5 10 12.5 15 7.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function OverviewPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-full border border-ink/10 bg-mist/70 px-3 py-2 text-sm text-ink/75">
      <span className="font-semibold text-ink">{value}</span> {label}
    </div>
  );
}

function ContextGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="rounded-2xl border border-ink/8 bg-mist/55 p-4">
      <p className="text-sm font-semibold text-ink">{title}</p>
      <div className="mt-3 space-y-2 text-sm text-ink/75">{children}</div>
    </div>
  );
}

export function ResultPanel({ result }: { result: AnalysisResponse }) {
  const contextItemCount =
    result.vitals.length + result.allergies.length + result.surgeries.length + result.risk_factors.length;
  const hasContext =
    result.vitals.length > 0 ||
    result.allergies.length > 0 ||
    result.surgeries.length > 0 ||
    result.risk_factors.length > 0;
  const showTransparency =
    result.meta.partial_data ||
    result.meta.judge.status !== "passed" ||
    result.meta.judge.issues.length > 0 ||
    result.meta.sources.length > 0;

  return (
    <section id="results" className="scroll-mt-6 space-y-4">
      <div className="grid gap-4 xl:grid-cols-[1.35fr_0.65fr]">
        <article className="rounded-[2rem] border border-white/60 bg-white/92 p-5 shadow-panel backdrop-blur md:p-6">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-xs uppercase tracking-[0.32em] text-ink/45">Plain-Language Breakdown</p>
              <h2 className="mt-2 font-serif text-3xl text-ink md:text-[2.2rem]">What this report means</h2>
            </div>
            <div className="flex flex-wrap gap-2">
              {result.diagnoses.length > 0 && <OverviewPill label="active problems" value={String(result.diagnoses.length)} />}
              {result.medications.length > 0 && <OverviewPill label="medications" value={String(result.medications.length)} />}
              {result.labs.length > 0 && <OverviewPill label="labs" value={String(result.labs.length)} />}
              {contextItemCount > 0 && <OverviewPill label="context items" value={String(contextItemCount)} />}
            </div>
          </div>

          <p className="mt-4 text-base leading-7 text-ink/80">{result.summary}</p>

          {result.warnings.length > 0 && (
            <div className="mt-5 rounded-3xl bg-coral/10 p-4 text-sm text-ink">
              <p className="font-semibold text-coral">Important context</p>
              <ul className="mt-2 space-y-2">
                {result.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}

          {result.meta.partial_data && (
            <div className="mt-4 rounded-2xl border border-gold/30 bg-gold/12 px-4 py-3 text-sm text-ink/80">
              Some parts of this explanation were simplified because the result was incomplete or needed extra review.
            </div>
          )}
        </article>

        <article className="rounded-[2rem] border border-ink/12 bg-ink p-5 text-mist shadow-panel md:p-6">
          <p className="text-xs uppercase tracking-[0.32em] text-mist/55">Next Step</p>
          <h2 className="mt-2 font-serif text-3xl">Bring these questions</h2>
          <ol className="mt-4 space-y-2 text-sm leading-6 text-mist/92">
            {result.questions_for_doctor.map((question, index) => (
              <li key={question} className="rounded-2xl border border-white/10 bg-white/5 p-3">
                <span className="mr-2 text-mist/45">{index + 1}.</span>
                {question}
              </li>
            ))}
          </ol>

          <p className="mt-4 text-xs leading-6 text-mist/70">{result.disclaimer}</p>

          {showTransparency && (
            <details className="group mt-4 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-mist/84">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-3 font-semibold text-mist">
                <span>How this answer was checked</span>
                <span className="flex items-center gap-2 text-xs uppercase tracking-[0.22em] text-mist/60">
                  Review
                  <ChevronDownIcon className="h-4 w-4 transition-transform duration-200 group-open:rotate-180" />
                </span>
              </summary>
              <div className="mt-3 space-y-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-white/10 px-3 py-1 text-xs uppercase tracking-[0.22em] text-mist/80">
                    Automated review: {judgeLabel[result.meta.judge.status]}
                  </span>
                  {result.meta.sources.map((source) => (
                    <span
                      key={source}
                      className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-[0.22em] text-mist/70"
                    >
                      {source}
                    </span>
                  ))}
                </div>

                {result.meta.partial_data_reasons.length > 0 && (
                  <div>
                    <p className="font-semibold text-mist">Notes</p>
                    <ul className="mt-2 space-y-1 text-mist/74">
                      {result.meta.partial_data_reasons.map((reason) => (
                        <li key={reason}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {result.meta.judge.issues.length > 0 && (
                  <div>
                    <p className="font-semibold text-mist">Review issues</p>
                    <ul className="mt-2 space-y-1 text-mist/74">
                      {result.meta.judge.issues.map((issue) => (
                        <li key={issue}>{issue}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </details>
          )}
        </article>
      </div>

      {(result.medications.length > 0 || result.diagnoses.length > 0 || result.labs.length > 0 || hasContext) && (
        <div className={`grid gap-4 ${result.medications.length > 0 && (result.diagnoses.length > 0 || result.labs.length > 0 || hasContext) ? "xl:grid-cols-[1.08fr_0.92fr]" : ""}`}>
          {result.medications.length > 0 && (
            <article className="rounded-[2rem] border border-white/60 bg-white/92 p-5 shadow-panel md:p-6">
              <div className="flex items-center justify-between">
                <h3 className="font-serif text-[2rem] text-ink">Medications</h3>
                <span className="text-xs uppercase tracking-[0.28em] text-ink/45">{result.medications.length}</span>
              </div>

              <div className="mt-4 space-y-3">
                {result.medications.map((medication) => (
                  <details key={medication.name} className="group rounded-3xl border border-ink/8 bg-mist/65 p-4">
                    <summary className="flex cursor-pointer list-none flex-wrap items-start gap-3">
                      <div className="min-w-0 flex-1">
                        <p className="text-base font-semibold text-ink">{medication.name}</p>
                        <p className="mt-1 text-sm leading-6 text-ink/72">{medication.purpose}</p>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        <span className={`rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.22em] ${groundingTone[medication.grounding_status]}`}>
                          {groundingLabel[medication.grounding_status]}
                        </span>
                        <span className={`rounded-full px-3 py-1 text-[11px] uppercase tracking-[0.22em] ${medicationStatusTone[medication.status]}`}>
                          {medicationStatusLabel[medication.status]}
                        </span>
                        <span className="flex items-center gap-1 rounded-full border border-ink/10 px-3 py-1 text-[11px] uppercase tracking-[0.22em] text-ink/55">
                          Details
                          <ChevronDownIcon className="h-3.5 w-3.5 transition-transform duration-200 group-open:rotate-180" />
                        </span>
                      </div>
                    </summary>

                    <div className="mt-4 space-y-3 border-t border-ink/10 pt-4 text-sm text-ink/76">
                      <p className="text-xs uppercase tracking-[0.22em] text-ink/45">{medication.grounding_note}</p>

                      {medication.cautions.length > 0 && (
                        <div className="rounded-2xl bg-gold/16 p-3">
                          <p className="font-semibold text-ink">Label cautions</p>
                          <ul className="mt-2 space-y-2">
                            {medication.cautions.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {medication.common_side_effects.length > 0 && (
                        <div>
                          <p className="font-semibold text-ink">Common side effects</p>
                          <ul className="mt-2 space-y-2">
                            {medication.common_side_effects.map((item) => (
                              <li key={item}>{item}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {medication.evidence.length > 0 && (
                        <div className="rounded-2xl border border-ink/10 bg-white/55 p-3">
                          <p className="text-xs uppercase tracking-[0.22em] text-ink/45">Evidence from label</p>
                          <ul className="mt-2 space-y-2">
                            {medication.evidence.slice(0, 2).map((item) => (
                              <li key={item.chunk_id}>
                                <span className="font-semibold text-ink">{item.label_section.replaceAll("_", " ")}</span>: {item.snippet}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </details>
                ))}
              </div>
            </article>
          )}

          {(result.diagnoses.length > 0 || result.labs.length > 0 || hasContext) && (
            <div className="grid gap-4">
              {result.diagnoses.length > 0 && (
                <article className="rounded-[2rem] border border-white/60 bg-white/92 p-5 shadow-panel md:p-6">
                  <div className="flex items-center justify-between">
                    <h3 className="font-serif text-[2rem] text-ink">Active problems</h3>
                    <span className="text-xs uppercase tracking-[0.28em] text-ink/45">{result.diagnoses.length}</span>
                  </div>
                  <div className="mt-4 space-y-3">
                    {result.diagnoses.map((diagnosis) => (
                      <div key={diagnosis.term} className="rounded-2xl border border-ink/8 bg-mist/65 p-4">
                        <p className="text-base font-semibold text-ink">{diagnosis.term}</p>
                        <p className="mt-2 text-sm leading-6 text-ink/75">{diagnosis.plain_language}</p>
                      </div>
                    ))}
                  </div>
                </article>
              )}

              {result.labs.length > 0 && (
                <article className="rounded-[2rem] border border-white/60 bg-white/92 p-5 shadow-panel md:p-6">
                  <div className="flex items-center justify-between">
                    <h3 className="font-serif text-[2rem] text-ink">Lab results</h3>
                    <span className="text-xs uppercase tracking-[0.28em] text-ink/45">{result.labs.length}</span>
                  </div>
                  <div className="mt-4 space-y-3">
                    {result.labs.map((lab) => (
                      <div key={lab.name} className="rounded-2xl border border-ink/8 bg-mist/65 p-4">
                        <div className="flex flex-wrap items-start justify-between gap-3">
                          <div>
                            <p className="text-base font-semibold text-ink">{lab.name}</p>
                            <p className="mt-1 text-sm text-ink/65">
                              {lab.value} {lab.unit}
                              {lab.reference_range ? ` - ref ${lab.reference_range}` : ""}
                            </p>
                          </div>
                          <span className={`rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.22em] ${statusTone[lab.status]}`}>
                            {lab.status}
                          </span>
                        </div>
                        <p className="mt-3 text-sm leading-6 text-ink/75">{lab.explanation}</p>
                      </div>
                    ))}
                  </div>
                </article>
              )}

              {hasContext && (
                <article className="rounded-[2rem] border border-white/60 bg-white/92 p-5 shadow-panel md:p-6">
                  <h3 className="font-serif text-[2rem] text-ink">From the report</h3>
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {result.vitals.length > 0 && (
                      <ContextGroup title="Vitals">
                        {result.vitals.map((vital) => (
                          <p key={vital.name}>
                            <span className="font-semibold text-ink">{vital.name}</span>: {vital.value} {vital.unit}
                          </p>
                        ))}
                      </ContextGroup>
                    )}

                    {result.allergies.length > 0 && (
                      <ContextGroup title="Allergies">
                        {result.allergies.map((allergy) => (
                          <p key={allergy.substance}>
                            <span className="font-semibold text-ink">{allergy.substance}</span>
                            {allergy.reaction ? `: ${allergy.reaction}` : ""}
                          </p>
                        ))}
                      </ContextGroup>
                    )}

                    {result.surgeries.length > 0 && (
                      <ContextGroup title="Surgeries">
                        {result.surgeries.map((surgery) => (
                          <p key={`${surgery.procedure}-${surgery.timing}`}>
                            <span className="font-semibold text-ink">{surgery.procedure}</span>
                            {[surgery.timing, surgery.reason].filter(Boolean).length > 0
                              ? ` - ${[surgery.timing, surgery.reason].filter(Boolean).join(" - ")}`
                              : ""}
                          </p>
                        ))}
                      </ContextGroup>
                    )}

                    {result.risk_factors.length > 0 && (
                      <ContextGroup title="Risk & history context">
                        {result.risk_factors.map((factor) => (
                          <div key={factor.factor}>
                            <p className="font-semibold text-ink">{factor.factor}</p>
                            <p className="mt-1">{factor.plain_language}</p>
                          </div>
                        ))}
                      </ContextGroup>
                    )}
                  </div>
                </article>
              )}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
