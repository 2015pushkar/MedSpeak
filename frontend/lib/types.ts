export type DocumentType = "lab" | "medication" | "diagnosis" | "mixed" | "unknown";
export type LabStatus = "low" | "normal" | "high" | "unknown";
export type GroundingStatus = "rag" | "openfda_live" | "text_only";
export type MedicationStatus = "current" | "historical" | "otc_prn" | "unclear";
export type ProcessingSource = "llm" | "heuristic" | "template";
export type JudgeStatus = "skipped" | "passed" | "failed" | "unavailable";

export type LabResult = {
  name: string;
  value: string;
  unit: string;
  reference_range: string;
  status: LabStatus;
  explanation: string;
};

export type MedicationEvidence = {
  source: string;
  label_section: string;
  chunk_id: string;
  snippet: string;
};

export type MedicationResult = {
  name: string;
  purpose: string;
  common_side_effects: string[];
  cautions: string[];
  fda_enriched: boolean;
  grounding_status: GroundingStatus;
  status: MedicationStatus;
  grounding_note: string;
  evidence: MedicationEvidence[];
};

export type DiagnosisResult = {
  term: string;
  plain_language: string;
};

export type VitalResult = {
  name: string;
  value: string;
  unit: string;
};

export type AllergyResult = {
  substance: string;
  reaction: string;
};

export type SurgeryResult = {
  procedure: string;
  timing: string;
  reason: string;
};

export type RiskFactorResult = {
  factor: string;
  plain_language: string;
};

export type ProcessingTrace = {
  classifier: ProcessingSource;
  medications: ProcessingSource;
  diagnoses: ProcessingSource;
  synthesis: ProcessingSource;
};

export type JudgeMeta = {
  status: JudgeStatus;
  model: string;
  issues: string[];
  blocked_sections: string[];
};

export type AnalysisMeta = {
  rate_limit_remaining: number;
  daily_limit: number;
  rate_limit_reset_at: string;
  partial_data: boolean;
  partial_data_reasons: string[];
  fallback_used: boolean;
  sources: string[];
  processing_trace: ProcessingTrace;
  judge: JudgeMeta;
};

export type AnalysisResponse = {
  document_type: DocumentType;
  summary: string;
  warnings: string[];
  labs: LabResult[];
  medications: MedicationResult[];
  diagnoses: DiagnosisResult[];
  vitals: VitalResult[];
  allergies: AllergyResult[];
  surgeries: SurgeryResult[];
  risk_factors: RiskFactorResult[];
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
