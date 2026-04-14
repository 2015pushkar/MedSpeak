import { AnalysisWorkbench } from "@/components/AnalysisWorkbench";

export default function HomePage() {
  return (
    <main className="min-h-screen px-4 py-5 md:px-8 md:py-6">
      <div className="mx-auto max-w-7xl space-y-5">
        <section className="rounded-[2rem] border border-white/60 bg-white/78 px-5 py-4 shadow-panel backdrop-blur md:px-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="min-w-0">
              <p className="text-xs uppercase tracking-[0.38em] text-ink/45">MedSpeak</p>
              <h1 className="mt-2 font-serif text-3xl leading-tight text-ink md:text-4xl">
                Medical documents, in plain language.
              </h1>
              <p className="mt-2 max-w-3xl text-sm leading-6 text-ink/72 md:text-base">
                Paste report text or upload a PDF to get a concise summary, medication cautions, and better follow-up
                questions.
              </p>
            </div>

            <div className="max-w-md rounded-2xl border border-leaf/18 bg-leaf/6 px-4 py-3 text-sm leading-6 text-ink/72">
              <span className="font-semibold text-ink">Educational only.</span> MedSpeak does not provide medical advice,
              diagnosis, or treatment.
            </div>
          </div>
        </section>

        <AnalysisWorkbench />
      </div>
    </main>
  );
}
