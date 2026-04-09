import { AnalysisWorkbench } from "@/components/AnalysisWorkbench";

export default function HomePage() {
  return (
    <main className="min-h-screen px-4 py-8 md:px-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <section className="grid gap-6 rounded-[2.5rem] border border-white/60 bg-white/70 px-6 py-8 shadow-panel backdrop-blur md:px-10 lg:grid-cols-[1.1fr_0.9fr]">
          <div>
            <p className="text-xs uppercase tracking-[0.4em] text-ink/45">MedSpeak</p>
            <h1 className="mt-4 max-w-3xl font-serif text-5xl leading-tight text-ink md:text-6xl">
              Medical documents, rewritten for an actual human reader.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-8 text-ink/75">
              Upload a PDF or paste report text, then get a plain-language summary, lab callouts, medication cautions,
              and better questions for your next appointment.
            </p>
          </div>

          <div className="rounded-[2rem] border border-ink/10 bg-mist/80 p-6">
            <p className="text-xs uppercase tracking-[0.35em] text-ink/45">Disclaimer</p>
            <p className="mt-4 text-base leading-8 text-ink/75">
              MedSpeak is an educational tool. It does not provide medical advice, diagnosis, or treatment. Always
              consult your healthcare provider for medical decisions.
            </p>
          </div>
        </section>

        <AnalysisWorkbench />
      </div>
    </main>
  );
}

