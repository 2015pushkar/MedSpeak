import type { AnalysisResponse, LabResult, LabStatus } from "@/lib/types";

const statusTone: Record<LabStatus, string> = {
  low: "bg-coral/15 text-coral",
  normal: "bg-leaf/15 text-leaf",
  high: "bg-gold/20 text-ink",
  unknown: "bg-ink/10 text-ink/70",
};

function LabRangeIndicator({ lab }: { lab: LabResult }) {
  const markerPosition =
    lab.status === "low" ? "left-[10%]" : lab.status === "high" ? "left-[82%]" : lab.status === "normal" ? "left-1/2" : "left-[50%]";
  const markerColor =
    lab.status === "low" ? "bg-coral" : lab.status === "high" ? "bg-gold" : lab.status === "normal" ? "bg-leaf" : "bg-ink/40";

  return (
    <div className="space-y-2">
      <div className="relative h-3 rounded-full bg-gradient-to-r from-coral/55 via-leaf/55 to-gold/55">
        <span className={`absolute top-1/2 h-5 w-5 -translate-y-1/2 rounded-full border-2 border-white ${markerColor} ${markerPosition}`} />
      </div>
      <div className="flex justify-between text-xs text-ink/60">
        <span>Low</span>
        <span>Range</span>
        <span>High</span>
      </div>
    </div>
  );
}

export function ResultPanel({ result }: { result: AnalysisResponse }) {
  return (
    <section id="results" className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-[1.3fr_0.7fr]">
        <article className="rounded-[2rem] border border-white/60 bg-white/90 p-6 shadow-panel backdrop-blur">
          <p className="text-xs uppercase tracking-[0.35em] text-ink/45">Plain-Language Breakdown</p>
          <h2 className="mt-3 font-serif text-3xl text-ink">What this report is saying</h2>
          <p className="mt-4 text-base leading-7 text-ink/80">{result.summary}</p>
          {result.warnings.length > 0 && (
            <div className="mt-6 rounded-3xl bg-coral/12 p-4 text-sm text-ink">
              <p className="font-semibold text-coral">Important context</p>
              <ul className="mt-2 space-y-2">
                {result.warnings.map((warning) => (
                  <li key={warning}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </article>

        <article className="rounded-[2rem] border border-white/60 bg-ink p-6 text-mist shadow-panel">
          <p className="text-xs uppercase tracking-[0.35em] text-mist/50">Doctor Visit Prep</p>
          <h2 className="mt-3 font-serif text-3xl">Questions to bring</h2>
          <ul className="mt-5 space-y-3 text-sm leading-6 text-mist/90">
            {result.questions_for_doctor.map((question) => (
              <li key={question} className="rounded-2xl border border-white/10 bg-white/5 p-3">
                {question}
              </li>
            ))}
          </ul>
        </article>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <article className="rounded-[2rem] border border-white/60 bg-white/90 p-6 shadow-panel">
          <div className="flex items-center justify-between">
            <h3 className="font-serif text-2xl text-ink">Lab results</h3>
            <span className="text-xs uppercase tracking-[0.3em] text-ink/45">{result.labs.length}</span>
          </div>
          <div className="mt-5 space-y-4">
            {result.labs.length > 0 ? (
              result.labs.map((lab) => (
                <div key={lab.name} className="rounded-3xl border border-ink/8 bg-mist/70 p-4">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <p className="text-lg font-semibold text-ink">{lab.name}</p>
                      <p className="text-sm text-ink/60">
                        {lab.value} {lab.unit}
                        {lab.reference_range ? ` • ref ${lab.reference_range}` : ""}
                      </p>
                    </div>
                    <span className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.24em] ${statusTone[lab.status]}`}>
                      {lab.status}
                    </span>
                  </div>
                  <div className="mt-4">
                    <LabRangeIndicator lab={lab} />
                  </div>
                  <p className="mt-4 text-sm leading-6 text-ink/75">{lab.explanation}</p>
                </div>
              ))
            ) : (
              <p className="rounded-3xl border border-dashed border-ink/15 bg-mist/60 p-4 text-sm text-ink/65">
                No lab-style values were extracted from this document.
              </p>
            )}
          </div>
        </article>

        <article className="rounded-[2rem] border border-white/60 bg-white/90 p-6 shadow-panel">
          <div className="flex items-center justify-between">
            <h3 className="font-serif text-2xl text-ink">Medications</h3>
            <span className="text-xs uppercase tracking-[0.3em] text-ink/45">{result.medications.length}</span>
          </div>
          <div className="mt-5 space-y-4">
            {result.medications.length > 0 ? (
              result.medications.map((medication) => (
                <div key={medication.name} className="rounded-3xl border border-ink/8 bg-mist/70 p-4">
                  <div className="flex items-center justify-between gap-4">
                    <p className="text-lg font-semibold text-ink">{medication.name}</p>
                    <span className="rounded-full bg-ink/8 px-3 py-1 text-xs uppercase tracking-[0.24em] text-ink/60">
                      {medication.fda_enriched ? "OpenFDA" : "Text only"}
                    </span>
                  </div>
                  <p className="mt-3 text-sm leading-6 text-ink/75">{medication.purpose}</p>
                  {medication.common_side_effects.length > 0 && (
                    <div className="mt-4">
                      <p className="text-xs uppercase tracking-[0.24em] text-ink/45">Common side effects</p>
                      <ul className="mt-2 space-y-2 text-sm text-ink/75">
                        {medication.common_side_effects.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {medication.cautions.length > 0 && (
                    <div className="mt-4 rounded-2xl bg-gold/18 p-3 text-sm text-ink">
                      <p className="font-semibold">Label cautions</p>
                      <ul className="mt-2 space-y-2">
                        {medication.cautions.map((item) => (
                          <li key={item}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              ))
            ) : (
              <p className="rounded-3xl border border-dashed border-ink/15 bg-mist/60 p-4 text-sm text-ink/65">
                No medication names were confidently extracted from this document.
              </p>
            )}
          </div>
        </article>

        <article className="rounded-[2rem] border border-white/60 bg-white/90 p-6 shadow-panel">
          <div className="flex items-center justify-between">
            <h3 className="font-serif text-2xl text-ink">Diagnoses</h3>
            <span className="text-xs uppercase tracking-[0.3em] text-ink/45">{result.diagnoses.length}</span>
          </div>
          <div className="mt-5 space-y-4">
            {result.diagnoses.length > 0 ? (
              result.diagnoses.map((diagnosis) => (
                <div key={diagnosis.term} className="rounded-3xl border border-ink/8 bg-mist/70 p-4">
                  <p className="text-lg font-semibold text-ink">{diagnosis.term}</p>
                  <p className="mt-3 text-sm leading-6 text-ink/75">{diagnosis.plain_language}</p>
                </div>
              ))
            ) : (
              <p className="rounded-3xl border border-dashed border-ink/15 bg-mist/60 p-4 text-sm text-ink/65">
                No diagnosis terms were confidently extracted from this document.
              </p>
            )}
          </div>
        </article>
      </div>

      <div className="rounded-[2rem] border border-white/60 bg-white/85 p-5 text-sm leading-6 text-ink/70 shadow-panel">
        <p>{result.disclaimer}</p>
        {result.meta.partial_data && (
          <div className="mt-3">
            <p className="font-semibold text-ink">Partial data notes</p>
            <ul className="mt-2 space-y-1">
              {result.meta.partial_data_reasons.map((reason) => (
                <li key={reason}>{reason}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

