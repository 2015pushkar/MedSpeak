import { expect, test } from "@playwright/test";

const rateStatus = {
  remaining: 5,
  daily_limit: 5,
  reset_at: "2099-01-01T00:00:00+00:00",
};

const analysisPayload = {
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
      grounding_status: "rag",
      status: "current",
      grounding_note: "grounded from local corpus",
      evidence: [
        {
          source: "chromadb",
          label_section: "indications_and_usage",
          chunk_id: "lisinopril-indications-0",
          snippet: "Lisinopril is used to treat hypertension.",
        },
      ],
    },
  ],
  diagnoses: [
    {
      term: "Hypertension",
      plain_language: "High blood pressure.",
    },
  ],
  vitals: [
    {
      name: "Blood Pressure",
      value: "168/98",
      unit: "",
    },
  ],
  allergies: [
    {
      substance: "Penicillin",
      reaction: "Rash and hives.",
    },
  ],
  surgeries: [
    {
      procedure: "Total abdominal hysterectomy and bilateral oophorectomy",
      timing: "1998",
      reason: "uterine fibroids",
    },
  ],
  risk_factors: [
    {
      factor: "Family history of premature CAD",
      plain_language: "Early heart disease in the family can matter when chest pain is evaluated.",
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
    processing_trace: {
      classifier: "llm",
      medications: "llm",
      diagnoses: "llm",
      synthesis: "template",
    },
    judge: {
      status: "passed",
      model: "gpt-4o-mini",
      issues: [],
      blocked_sections: [],
    },
  },
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/rate-status", async (route) => {
    await route.fulfill({ json: rateStatus });
  });
});

test("analyzes pasted text", async ({ page }) => {
  await page.route("**/api/analyze", async (route) => {
    await route.fulfill({ json: analysisPayload });
  });

  await page.goto("/");
  await page.getByLabel("Paste report text").fill("Glucose: 142 mg/dL (70-100)");
  await page.getByRole("button", { name: "Explain This Report" }).click();

  await expect(page.getByText("What this report means")).toBeVisible();
  await expect(page.getByText("Glucose", { exact: true })).toBeVisible();
  await expect(page.getByText("Lisinopril", { exact: true })).toBeVisible();
  await expect(page.getByText("RAG grounded")).toBeVisible();
  await expect(page.getByText("Bring these questions")).toBeVisible();
  await expect(page.getByText("How this answer was checked")).toBeVisible();
});

test("analyzes an uploaded pdf", async ({ page }) => {
  await page.route("**/api/analyze", async (route) => {
    await route.fulfill({ json: analysisPayload });
  });

  await page.goto("/");
  await page.getByRole("button", { name: "Upload PDF" }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: "lab_report.pdf",
    mimeType: "application/pdf",
    buffer: Buffer.from("%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"),
  });
  await page.getByRole("button", { name: "Explain This Report" }).click();

  await expect(page.getByText("What this report means")).toBeVisible();
  await expect(page.getByText("Hypertension", { exact: true })).toBeVisible();
  await expect(page.getByText("Penicillin", { exact: true })).toBeVisible();
});
