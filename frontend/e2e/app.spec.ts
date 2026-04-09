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

  await expect(page.getByText("What this report is saying")).toBeVisible();
  await expect(page.getByText("Glucose", { exact: true })).toBeVisible();
  await expect(page.getByText("Lisinopril", { exact: true })).toBeVisible();
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

  await expect(page.getByText("What this report is saying")).toBeVisible();
  await expect(page.getByText("Hypertension", { exact: true })).toBeVisible();
});
