export type DocumentType = "lab" | "medication" | "diagnosis" | "mixed" | "unknown";
export type LabStatus = "low" | "normal" | "high" | "unknown";

export type LabResult = {
  name: string;
  value: string;
  unit: string;
  reference_range: string;
  status: LabStatus;
  explanation: string;
};

export type MedicationResult = {
  name: string;
  purpose: string;
  common_side_effects: string[];
  cautions: string[];
  fda_enriched: boolean;
};

export type DiagnosisResult = {
  term: string;
  plain_language: string;
};

export type AnalysisMeta = {
  rate_limit_remaining: number;
  daily_limit: number;
  rate_limit_reset_at: string;
  partial_data: boolean;
  partial_data_reasons: string[];
  fallback_used: boolean;
  sources: string[];
};

export type AnalysisResponse = {
  document_type: DocumentType;
  summary: string;
  warnings: string[];
  labs: LabResult[];
  medications: MedicationResult[];
  diagnoses: DiagnosisResult[];
  questions_for_doctor: string[];
  disclaimer: string;
  meta: AnalysisMeta;
};

export type RateStatus = {
  remaining: number;
  daily_limit: number;
  reset_at: string;
};

export type ApiErrorShape = {
  code: string;
  message: string;
  details?: Record<string, unknown>;
};

