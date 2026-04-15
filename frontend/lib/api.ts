import type { AnalysisResponse, ApiErrorShape, RateStatus } from "@/lib/types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    cache: "no-store",
  });
  const payload = (await response.json().catch(() => null)) as T | ApiErrorShape | null;
  if (!response.ok) {
    throw (
      payload ?? {
        code: "unknown_error",
        message: "The request failed before a structured error could be returned.",
      }
    );
  }
  return payload as T;
}

export function getRateStatus(): Promise<RateStatus> {
  return request<RateStatus>("/api/rate-status");
}

export function analyzeDocument(formData: FormData): Promise<AnalysisResponse> {
  return request<AnalysisResponse>("/api/analyze", {
    method: "POST",
    body: formData,
  });
}
