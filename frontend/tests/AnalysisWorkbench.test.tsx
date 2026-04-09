import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { AnalysisWorkbench } from "@/components/AnalysisWorkbench";
import type { AnalysisResponse, RateStatus } from "@/lib/types";

const rateStatus: RateStatus = {
  remaining: 5,
  daily_limit: 5,
  reset_at: "2099-01-01T00:00:00+00:00",
};

const mixedResult: AnalysisResponse = {
  document_type: "mixed",
  summary: "It includes abnormal labs, medications, and diagnosis terms in one discharge summary.",
  warnings: ["Some lab values appear outside the listed reference range."],
  labs: [
    {
      name: "Glucose",
      value: "142",
      unit: "mg/dL",
      reference_range: "70-100",
      status: "high",
      explanation: "Glucose is above the listed reference range.",
    },
  ],
  medications: [
    {
      name: "Lisinopril",
      purpose: "May be used for blood pressure support.",
      common_side_effects: ["Dizziness."],
      cautions: ["Monitor blood pressure changes."],
      fda_enriched: true,
    },
  ],
  diagnoses: [
    {
      term: "Hypertension",
      plain_language: "High blood pressure.",
    },
  ],
  questions_for_doctor: ["What might explain my high glucose result?"],
  disclaimer: "Educational use only.",
  meta: {
    rate_limit_remaining: 4,
    daily_limit: 5,
    rate_limit_reset_at: "2099-01-01T00:00:00+00:00",
    partial_data: false,
    partial_data_reasons: [],
    fallback_used: true,
    sources: ["openfda"],
  },
};

function mockResponse(payload: unknown, ok = true) {
  return {
    ok,
    json: async () => payload,
  } as Response;
}

describe("AnalysisWorkbench", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("validates empty text submissions", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce(mockResponse(rateStatus));
    const user = userEvent.setup();

    render(<AnalysisWorkbench />);
    await screen.findByText(/5 of 5 free analyses left today/i);
    await user.click(screen.getByRole("button", { name: /explain this report/i }));

    expect(screen.getByText(/paste report text before starting an analysis/i)).toBeInTheDocument();
  });

  it("shows the loading state while analysis is in progress", async () => {
    let resolveAnalysis: ((value: Response) => void) | undefined;
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse(rateStatus))
      .mockImplementationOnce(
        () =>
          new Promise<Response>((resolve) => {
            resolveAnalysis = resolve;
          }),
      );

    const user = userEvent.setup();
    render(<AnalysisWorkbench />);

    await screen.findByText(/5 of 5 free analyses left today/i);
    await user.type(screen.getByLabelText(/paste report text/i), "Glucose: 142 mg/dL (70-100)");
    await user.click(screen.getByRole("button", { name: /explain this report/i }));

    expect(screen.getByText(/translating the report into everyday language/i)).toBeInTheDocument();

    resolveAnalysis?.(mockResponse(mixedResult));
    await waitFor(() => expect(screen.getByText(/what this report is saying/i)).toBeInTheDocument());
  });

  it("renders the quota exhausted state", async () => {
    vi.spyOn(global, "fetch").mockResolvedValueOnce(
      mockResponse({
        remaining: 0,
        daily_limit: 5,
        reset_at: "2099-01-01T00:00:00+00:00",
      }),
    );
    render(<AnalysisWorkbench />);

    expect(await screen.findByText(/0 of 5 free analyses left today/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /explain this report/i })).toBeDisabled();
  });

  it("renders backend errors", async () => {
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse(rateStatus))
      .mockResolvedValueOnce(
        mockResponse(
          {
            code: "rate_limit_exceeded",
            message: "You have used all free analyses for today.",
            details: {
              daily_limit: 5,
              reset_at: "2099-01-01T00:00:00+00:00",
            },
          },
          false,
        ),
      );

    const user = userEvent.setup();
    render(<AnalysisWorkbench />);

    await screen.findByText(/5 of 5 free analyses left today/i);
    await user.type(screen.getByLabelText(/paste report text/i), "Glucose: 142 mg/dL (70-100)");
    await user.click(screen.getByRole("button", { name: /explain this report/i }));

    expect(await screen.findByText(/you have used all free analyses for today/i)).toBeInTheDocument();
  });

  it("renders mixed analysis results", async () => {
    vi.spyOn(global, "fetch")
      .mockResolvedValueOnce(mockResponse(rateStatus))
      .mockResolvedValueOnce(mockResponse(mixedResult));

    const user = userEvent.setup();
    render(<AnalysisWorkbench />);

    await screen.findByText(/5 of 5 free analyses left today/i);
    await user.type(screen.getByLabelText(/paste report text/i), "Mixed report");
    await user.click(screen.getByRole("button", { name: /explain this report/i }));

    expect(await screen.findByText(/what this report is saying/i)).toBeInTheDocument();
    expect(screen.getByText("Glucose")).toBeInTheDocument();
    expect(screen.getByText("Lisinopril")).toBeInTheDocument();
    expect(screen.getByText("Hypertension")).toBeInTheDocument();
    expect(screen.getByText(/what might explain my high glucose result/i)).toBeInTheDocument();
  });
});

