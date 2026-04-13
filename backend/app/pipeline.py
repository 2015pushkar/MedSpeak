from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from app.config import Settings
from app.prompts.agent_prompts import (
    CLASSIFIER_PROMPT,
    DIAGNOSIS_PROMPT,
    LAB_ANALYST_PROMPT,
    MEDICATION_PROMPT,
    SYNTHESIS_PROMPT,
)
from app.schemas import (
    AnalysisMeta,
    AnalysisResponse,
    DiagnosisResult,
    DocumentType,
    LabResult,
    MedicationEvidence,
    MedicationResult,
)
from app.services.medication_rag import MedicationRAGService
from app.services.safety import SafetyService
from app.state import PipelineState
from app.tools.openfda import MedicationEnrichment, OpenFDATool
from app.utils.lab_ranges import (
    build_lab_explanation,
    format_reference_range,
    normalize_lab_name,
    resolve_reference_range,
)

DISCLAIMER = (
    "MedSpeak is an educational tool. It does not provide medical advice, diagnosis, or treatment. "
    "Always consult your healthcare provider for medical decisions."
)

KNOWN_DIAGNOSES = {
    "hypertension": "High blood pressure. It means blood is pushing too strongly against the walls of your arteries.",
    "type 2 diabetes": "A condition where the body has trouble managing blood sugar over time.",
    "diabetes mellitus": "A condition where the body has trouble controlling blood sugar.",
    "hyperlipidemia": "Higher-than-desired fats such as cholesterol or triglycerides in the blood.",
    "anemia": "A lower-than-expected amount of healthy red blood cells or hemoglobin, which can reduce oxygen delivery.",
    "hypokalemia": "A lower-than-normal potassium level.",
    "pneumonia": "An infection or inflammation affecting the lungs.",
    "chronic kidney disease": "Long-term kidney damage or reduced kidney function.",
    "asthma": "A condition that can narrow the airways and make breathing harder.",
}


@dataclass(frozen=True)
class CompletionResult:
    payload: dict[str, Any] | None
    failure_reason: str | None = None


class MedicalPipeline:
    def __init__(self, settings: Settings, openfda_tool: OpenFDATool) -> None:
        self.settings = settings
        self.openfda_tool = openfda_tool
        self.client = self._build_client()
        self.medication_rag = MedicationRAGService.from_settings(settings, self.client)
        self.safety = SafetyService()
        self.graph = self._build_graph()

    def _build_client(self) -> AsyncOpenAI | None:
        if not self.settings.openai_enabled:
            return None
        return AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url or None,
        )

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        graph.add_node("classifier", self._classify_node)
        graph.add_node("lab_agent", self._lab_agent_node)
        graph.add_node("medication_agent", self._medication_agent_node)
        graph.add_node("diagnosis_agent", self._diagnosis_agent_node)
        graph.add_node("synthesis", self._synthesis_node)

        graph.add_edge(START, "classifier")
        graph.add_conditional_edges("classifier", self._route_after_classifier)
        graph.add_edge("lab_agent", "synthesis")
        graph.add_edge("medication_agent", "synthesis")
        graph.add_edge("diagnosis_agent", "synthesis")
        graph.add_edge("synthesis", END)
        return graph.compile()

    async def analyze(
        self,
        *,
        text: str,
        rate_limit_remaining: int,
        daily_limit: int,
        reset_at: str,
        partial_data_reasons: list[str] | None = None,
    ) -> AnalysisResponse:
        state: PipelineState = {
            "input_text": text,
            "warnings": [],
            "partial_data_reasons": partial_data_reasons or [],
            "fallback_used": False,
            "sources": [],
        }
        result = await self.graph.ainvoke(state)
        summary = result.get(
            "summary",
            "This document was processed, but MedSpeak could not build a fuller explanation from the available text.",
        )
        warnings = self._dedupe(result.get("warnings", []))
        partial_reasons = self._dedupe(result.get("partial_data_reasons", []))
        sources = self._dedupe(result.get("sources", []))
        questions = self._dedupe(result.get("questions_for_doctor", []))[:5]

        safety_result = await self.safety.enforce(
            summary=summary,
            warnings=warnings,
            questions_for_doctor=questions or ["What are the most important next steps or follow-up questions from this report?"],
            client=self.client,
            model=self.settings.openai_analyst_model,
        )
        summary = safety_result.summary
        warnings = self._dedupe(safety_result.warnings)
        questions = self._dedupe(safety_result.questions_for_doctor)[:5]
        partial_reasons = self._dedupe([*partial_reasons, *safety_result.partial_data_reasons])

        return AnalysisResponse(
            document_type=result.get("document_type", "unknown"),
            summary=summary,
            warnings=warnings,
            labs=result.get("labs", []),
            medications=result.get("medications", []),
            diagnoses=result.get("diagnoses", []),
            questions_for_doctor=questions
            or ["What are the most important next steps or follow-up questions from this report?"],
            disclaimer=DISCLAIMER,
            meta=AnalysisMeta(
                rate_limit_remaining=rate_limit_remaining,
                daily_limit=daily_limit,
                rate_limit_reset_at=reset_at,
                partial_data=bool(partial_reasons),
                partial_data_reasons=partial_reasons,
                fallback_used=bool(result.get("fallback_used", False) or safety_result.canned_response_used),
                sources=sources,
            ),
        )

    def _route_after_classifier(self, state: PipelineState) -> list[str]:
        targets = state.get("agent_targets", [])
        return targets or ["synthesis"]

    async def _classify_node(self, state: PipelineState) -> PipelineState:
        llm_result = await self._classify_with_llm(state["input_text"])
        if llm_result.payload:
            return {
                "document_type": llm_result.payload["document_type"],
                "agent_targets": llm_result.payload["agent_targets"],
            }
        document_type, targets = self._heuristic_classification(state["input_text"])
        payload: PipelineState = {
            "document_type": document_type,
            "agent_targets": targets,
            "fallback_used": True,
        }
        if llm_result.failure_reason:
            payload["partial_data_reasons"] = [llm_result.failure_reason]
        return payload

    async def _lab_agent_node(self, state: PipelineState) -> PipelineState:
        labs = self._extract_labs_heuristically(state["input_text"])
        partial_reasons: list[str] = []
        fallback_used = False
        if labs and self.client:
            explained, failure_reason = await self._enhance_labs_with_llm(labs)
            if explained:
                labs = explained
            else:
                fallback_used = True
                if failure_reason:
                    partial_reasons.append(failure_reason)
        payload: PipelineState = {"labs": labs}
        if fallback_used:
            payload["fallback_used"] = True
        if partial_reasons:
            payload["partial_data_reasons"] = partial_reasons
        return payload

    async def _medication_agent_node(self, state: PipelineState) -> PipelineState:
        medications, fallback_used, extraction_reasons = await self._extract_medications(state["input_text"])
        results: list[MedicationResult] = []
        partial_reasons = list(extraction_reasons)
        sources: list[str] = []

        for medication in medications:
            rag_grounding = await self.medication_rag.ground_medication(
                medication.name,
                medication.purpose or "This medication was mentioned in the document.",
                top_k=self.settings.retrieval_top_k,
            )
            if rag_grounding:
                results.append(
                    MedicationResult(
                        name=medication.name,
                        purpose=rag_grounding.purpose,
                        common_side_effects=rag_grounding.common_side_effects,
                        cautions=rag_grounding.cautions,
                        fda_enriched=True,
                        grounding_status="rag",
                        evidence=rag_grounding.evidence,
                    )
                )
                sources.append(rag_grounding.backend)
                continue

            retrieval = await self.medication_rag.retrieve(medication.name, top_k=self.settings.retrieval_top_k)
            if retrieval.partial_reason:
                partial_reasons.append(retrieval.partial_reason)
            try:
                enrichment = await self.openfda_tool.lookup(medication.name)
            except Exception:
                enrichment = MedicationEnrichment()
                partial_reasons.append(f"OpenFDA enrichment was unavailable for {medication.name}.")
            evidence = self._build_live_openfda_evidence(enrichment)
            merged = MedicationResult(
                name=medication.name,
                purpose=enrichment.purpose
                or medication.purpose
                or "This medication was mentioned in the document, but the report did not explain why it was prescribed.",
                common_side_effects=enrichment.common_side_effects,
                cautions=enrichment.cautions,
                fda_enriched=enrichment.fda_enriched,
                grounding_status="openfda_live" if enrichment.fda_enriched else "text_only",
                evidence=evidence if enrichment.fda_enriched else [],
            )
            if merged.fda_enriched:
                sources.append("openfda")
            results.append(merged)

        payload: PipelineState = {"medications": results}
        if partial_reasons:
            payload["partial_data_reasons"] = self._dedupe(partial_reasons)
        if sources:
            payload["sources"] = self._dedupe(sources)
        if fallback_used:
            payload["fallback_used"] = True
        return payload

    async def _diagnosis_agent_node(self, state: PipelineState) -> PipelineState:
        diagnoses, fallback_used, partial_reasons = await self._extract_diagnoses(state["input_text"])
        payload: PipelineState = {"diagnoses": diagnoses}
        if fallback_used:
            payload["fallback_used"] = True
        if partial_reasons:
            payload["partial_data_reasons"] = partial_reasons
        return payload

    async def _synthesis_node(self, state: PipelineState) -> PipelineState:
        warnings = self._build_warning_messages(state)
        questions = self._build_questions(state)
        summary = self._build_summary(state)

        if self.client:
            llm = await self._synthesize_with_llm(state, summary, warnings, questions)
            if llm.payload:
                return {
                    "summary": llm.payload.get("summary", summary),
                    "warnings": llm.payload.get("warnings", warnings),
                    "questions_for_doctor": llm.payload.get("questions_for_doctor", questions),
                }
            payload: PipelineState = {
                "summary": summary,
                "warnings": warnings,
                "questions_for_doctor": questions,
                "fallback_used": True,
            }
            if llm.failure_reason:
                payload["partial_data_reasons"] = [llm.failure_reason]
            return payload
        return {
            "summary": summary,
            "warnings": warnings,
            "questions_for_doctor": questions,
        }

    async def _classify_with_llm(self, text: str) -> CompletionResult:
        completion = await self._json_completion(
            model=self.settings.openai_classifier_model,
            prompt=CLASSIFIER_PROMPT,
            user_content=text[:6000],
            failure_label="Classifier model",
        )
        if not completion.payload:
            return completion
        document_type = completion.payload.get("document_type")
        targets = completion.payload.get("agent_targets") or []
        valid_targets = [target for target in targets if target in {"lab_agent", "medication_agent", "diagnosis_agent"}]
        if document_type not in {"lab", "medication", "diagnosis", "mixed", "unknown"}:
            return CompletionResult(payload=None, failure_reason="Classifier model returned an invalid routing decision.")
        return CompletionResult(payload={"document_type": document_type, "agent_targets": valid_targets})

    async def _enhance_labs_with_llm(self, labs: list[LabResult]) -> tuple[list[LabResult] | None, str | None]:
        completion = await self._json_completion(
            model=self.settings.openai_analyst_model,
            prompt=LAB_ANALYST_PROMPT,
            user_content=json.dumps({"labs": [lab.model_dump() for lab in labs]}),
            failure_label="Lab explanation model",
        )
        if not completion.payload:
            return None, completion.failure_reason
        explanations = {
            normalize_lab_name(item["name"]): item.get("explanation", "")
            for item in completion.payload.get("labs", [])
            if item.get("name")
        }
        enhanced: list[LabResult] = []
        for lab in labs:
            explanation = explanations.get(normalize_lab_name(lab.name)) or lab.explanation
            enhanced.append(lab.model_copy(update={"explanation": explanation}))
        return enhanced, None

    async def _extract_medications(self, text: str) -> tuple[list[MedicationResult], bool, list[str]]:
        if self.client:
            completion = await self._json_completion(
                model=self.settings.openai_analyst_model,
                prompt=MEDICATION_PROMPT,
                user_content=text[:6000],
                failure_label="Medication extraction model",
            )
            if completion.payload:
                medications = [
                    MedicationResult(
                        name=item["name"].strip(),
                        purpose=(item.get("purpose") or "").strip() or "This medication was mentioned in the document.",
                        common_side_effects=[],
                        cautions=[],
                        fda_enriched=False,
                        grounding_status="text_only",
                        evidence=[],
                    )
                    for item in completion.payload.get("medications", [])
                    if item.get("name")
                ]
                if medications:
                    return self._dedupe_medications(medications), False, []
            if completion.failure_reason:
                return self._extract_medications_heuristically(text), True, [completion.failure_reason]
        return self._extract_medications_heuristically(text), True, []

    async def _extract_diagnoses(self, text: str) -> tuple[list[DiagnosisResult], bool, list[str]]:
        if self.client:
            completion = await self._json_completion(
                model=self.settings.openai_analyst_model,
                prompt=DIAGNOSIS_PROMPT,
                user_content=text[:6000],
                failure_label="Diagnosis explanation model",
            )
            if completion.payload:
                diagnoses = [
                    DiagnosisResult(
                        term=item["term"].strip(),
                        plain_language=item.get("plain_language", "").strip()
                        or "This is a clinical term noted in the report. Ask your clinician how it applies to you.",
                    )
                    for item in completion.payload.get("diagnoses", [])
                    if item.get("term")
                ]
                if diagnoses:
                    return self._dedupe_diagnoses(diagnoses), False, []
            if completion.failure_reason:
                return self._extract_diagnoses_heuristically(text), True, [completion.failure_reason]
        return self._extract_diagnoses_heuristically(text), True, []

    async def _synthesize_with_llm(
        self,
        state: PipelineState,
        summary: str,
        warnings: list[str],
        questions: list[str],
    ) -> CompletionResult:
        completion = await self._json_completion(
            model=self.settings.openai_analyst_model,
            prompt=SYNTHESIS_PROMPT,
            user_content=json.dumps(
                {
                    "document_type": state.get("document_type", "unknown"),
                    "summary_seed": summary,
                    "warnings_seed": warnings,
                    "questions_seed": questions,
                    "labs": [item.model_dump() for item in state.get("labs", [])],
                    "medications": [item.model_dump() for item in state.get("medications", [])],
                    "diagnoses": [item.model_dump() for item in state.get("diagnoses", [])],
                }
            ),
            failure_label="Synthesis model",
        )
        if not completion.payload:
            return completion
        return CompletionResult(
            payload={
                "summary": completion.payload.get("summary", summary),
                "warnings": completion.payload.get("warnings", warnings),
                "questions_for_doctor": completion.payload.get("questions_for_doctor", questions),
            }
        )

    async def _json_completion(
        self,
        *,
        model: str,
        prompt: str,
        user_content: str,
        failure_label: str,
    ) -> CompletionResult:
        if not self.client:
            return CompletionResult(payload=None)

        last_error: str | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
                content = response.choices[0].message.content
                if not content:
                    last_error = f"{failure_label} returned an empty response."
                else:
                    try:
                        return CompletionResult(payload=json.loads(content))
                    except json.JSONDecodeError:
                        last_error = f"{failure_label} returned invalid JSON."
            except Exception:
                last_error = f"{failure_label} was unavailable after retries. MedSpeak used a deterministic fallback instead."
            if attempt < self.settings.llm_max_retries - 1:
                await asyncio.sleep(self.settings.llm_retry_base_delay_seconds * (2**attempt))
        return CompletionResult(payload=None, failure_reason=last_error)

    def _build_live_openfda_evidence(self, enrichment: MedicationEnrichment) -> list[MedicationEvidence]:
        evidence: list[MedicationEvidence] = []
        if enrichment.purpose:
            evidence.append(
                MedicationEvidence(
                    source="openfda_live",
                    label_section="indications_and_usage",
                    chunk_id="live-purpose",
                    snippet=enrichment.purpose,
                )
            )
        if enrichment.common_side_effects:
            evidence.append(
                MedicationEvidence(
                    source="openfda_live",
                    label_section="adverse_reactions",
                    chunk_id="live-side-effects",
                    snippet=enrichment.common_side_effects[0],
                )
            )
        elif enrichment.cautions:
            evidence.append(
                MedicationEvidence(
                    source="openfda_live",
                    label_section="warnings_and_cautions",
                    chunk_id="live-cautions",
                    snippet=enrichment.cautions[0],
                )
            )
        return evidence[:2]

    def _heuristic_classification(self, text: str) -> tuple[DocumentType, list[str]]:
        lab_score = len(re.findall(r"\b(glucose|potassium|sodium|hemoglobin|a1c|creatinine|wbc|platelets|bun)\b", text, re.IGNORECASE))
        lab_score += len(re.findall(r"\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|g/dl|k/ul|%)", text, re.IGNORECASE))
        med_score = len(re.findall(r"\b(?:mg|mcg|tablet|capsule|take|daily|bid|tid|prn|medication)\b", text, re.IGNORECASE))
        diagnosis_score = len(re.findall(r"\b(?:diagnosis|impression|assessment|history of|discharge diagnosis|icd-?10)\b", text, re.IGNORECASE))

        targets: list[str] = []
        if lab_score >= 2:
            targets.append("lab_agent")
        if med_score >= 2:
            targets.append("medication_agent")
        if diagnosis_score >= 1:
            targets.append("diagnosis_agent")

        if len(targets) > 1:
            return "mixed", targets
        if targets == ["lab_agent"]:
            return "lab", targets
        if targets == ["medication_agent"]:
            return "medication", targets
        if targets == ["diagnosis_agent"]:
            return "diagnosis", targets
        return "unknown", []

    def _extract_labs_heuristically(self, text: str) -> list[LabResult]:
        pattern = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z0-9\s/%+\-]{1,40}?)\s*[:\-]?\s*(?P<value>-?\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z/%0-9^]+)?\s*(?:\((?P<range>[^)]+)\))?",
            re.IGNORECASE,
        )
        results: list[LabResult] = []
        seen: set[str] = set()

        for line in text.splitlines():
            if not any(char.isdigit() for char in line):
                continue
            if re.search(r"\b(?:tablet|capsule|take|sig|daily|bid|tid|prn)\b", line, re.IGNORECASE):
                continue
            match = pattern.search(line.strip())
            if not match:
                continue
            name = match.group("name").strip(" -:")
            normalized_name = normalize_lab_name(name)
            if normalized_name in seen:
                continue
            range_text = match.group("range")
            if normalized_name not in {
                "glucose",
                "potassium",
                "sodium",
                "creatinine",
                "hemoglobin",
                "wbc",
                "platelets",
                "a1c",
                "bun",
                "cholesterol",
            } and not range_text:
                continue

            value = float(match.group("value"))
            interval = resolve_reference_range(name, range_text)
            status = "unknown"
            if interval:
                low, high = interval
                if value < low:
                    status = "low"
                elif value > high:
                    status = "high"
                else:
                    status = "normal"

            results.append(
                LabResult(
                    name=name,
                    value=f"{value:g}",
                    unit=(match.group("unit") or "").strip(),
                    reference_range=format_reference_range(name, range_text),
                    status=status,
                    explanation=build_lab_explanation(name, status),
                )
            )
            seen.add(normalized_name)
        return results

    def _extract_medications_heuristically(self, text: str) -> list[MedicationResult]:
        candidates: list[str] = []
        line_pattern = re.compile(
            r"(?P<name>[A-Za-z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+)?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|units?|ml|mL)\b",
            re.IGNORECASE,
        )
        section_pattern = re.compile(r"(?:medications?|discharge medications?)\s*[:\-]\s*(?P<items>.+)", re.IGNORECASE)

        for line in text.splitlines():
            section_match = section_pattern.search(line)
            if section_match:
                items = re.split(r"[,;]", section_match.group("items"))
                for item in items:
                    words = item.strip().split()
                    if words:
                        candidates.append(words[0])
            line_match = line_pattern.search(line)
            if line_match:
                candidates.append(line_match.group("name"))

        seen: set[str] = set()
        medications: list[MedicationResult] = []
        for candidate in candidates:
            cleaned = re.sub(r"\s+", " ", candidate).strip().title()
            normalized = cleaned.lower()
            if normalized in seen or normalized in {"glucose", "potassium", "sodium"}:
                continue
            medications.append(
                MedicationResult(
                    name=cleaned,
                    purpose="This medication was mentioned in the document.",
                    common_side_effects=[],
                    cautions=[],
                    fda_enriched=False,
                    grounding_status="text_only",
                    evidence=[],
                )
            )
            seen.add(normalized)
        return medications

    def _extract_diagnoses_heuristically(self, text: str) -> list[DiagnosisResult]:
        findings: list[str] = []
        label_pattern = re.compile(
            r"(?:diagnosis(?:es)?|assessment|impression|discharge diagnosis)\s*[:\-]\s*(?P<terms>.+)",
            re.IGNORECASE,
        )
        for line in text.splitlines():
            match = label_pattern.search(line)
            if match:
                findings.extend(part.strip() for part in re.split(r"[,;]", match.group("terms")) if part.strip())

        lowered_text = text.lower()
        for known in KNOWN_DIAGNOSES:
            if known in lowered_text:
                findings.append(known)

        diagnoses: list[DiagnosisResult] = []
        seen: set[str] = set()
        for finding in findings:
            cleaned = finding.strip()
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            diagnoses.append(
                DiagnosisResult(
                    term=cleaned.title(),
                    plain_language=KNOWN_DIAGNOSES.get(
                        normalized,
                        "This is a clinical term noted in the report. Ask your clinician what it means in your specific situation.",
                    ),
                )
            )
            seen.add(normalized)
        return diagnoses

    def _build_summary(self, state: PipelineState) -> str:
        labs = state.get("labs", [])
        medications = state.get("medications", [])
        diagnoses = state.get("diagnoses", [])
        document_type = state.get("document_type", "unknown")

        if document_type == "unknown" and not any([labs, medications, diagnoses]):
            return "MedSpeak could not confidently identify a familiar report pattern from the text that was provided."

        fragments = []
        if labs:
            abnormal = [lab.name for lab in labs if lab.status in {"low", "high"}]
            if abnormal:
                fragments.append(f"It includes {len(labs)} lab results, with out-of-range findings such as {', '.join(abnormal[:2])}.")
            else:
                fragments.append(f"It includes {len(labs)} lab results without obvious out-of-range values.")
        if medications:
            grounded = [medication for medication in medications if medication.grounding_status != "text_only"]
            if grounded:
                fragments.append(
                    f"It mentions {len(medications)} medication{'s' if len(medications) != 1 else ''}, including grounded label context for {len(grounded)}."
                )
            else:
                fragments.append(f"It mentions {len(medications)} medication{'s' if len(medications) != 1 else ''}.")
        if diagnoses:
            fragments.append(f"It references {len(diagnoses)} diagnosis term{'s' if len(diagnoses) != 1 else ''} in plain language.")
        return " ".join(fragments) or "This document was processed and organized into plain-language findings."

    def _build_warning_messages(self, state: PipelineState) -> list[str]:
        warnings = list(state.get("warnings", []))
        labs = state.get("labs", [])
        medications = state.get("medications", [])

        if any(lab.status in {"low", "high"} for lab in labs):
            warnings.append("Some lab values appear outside the listed reference range.")
        if any(medication.cautions for medication in medications):
            warnings.append("Medication label cautions were found. Ask a clinician how they apply to you.")
        if any(medication.grounding_status == "text_only" for medication in medications):
            warnings.append("Some medication explanations were text-only because grounded label context was unavailable.")
        if state.get("document_type") == "unknown":
            warnings.append("This text did not clearly match a standard lab, medication, or diagnosis report format.")
        warnings.extend(state.get("partial_data_reasons", []))
        return self._dedupe(warnings)

    def _build_questions(self, state: PipelineState) -> list[str]:
        questions: list[str] = []
        for lab in state.get("labs", []):
            if lab.status == "high":
                questions.append(f"What might explain my high {lab.name.lower()} result, and does it need follow-up?")
            elif lab.status == "low":
                questions.append(f"What might explain my low {lab.name.lower()} result, and should it be rechecked?")
        for medication in state.get("medications", []):
            questions.append(f"How does {medication.name} fit with the rest of this report, and do I need any monitoring?")
        for diagnosis in state.get("diagnoses", []):
            questions.append(f"What does {diagnosis.term.lower()} mean in my specific case?")
        if not questions:
            questions.append("What are the key takeaways from this report, and is any follow-up needed?")
        return self._dedupe(questions)

    def _dedupe(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        unique: list[str] = []
        for item in items:
            cleaned = item.strip()
            if not cleaned:
                continue
            normalized = cleaned.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(cleaned)
        return unique

    def _dedupe_medications(self, medications: list[MedicationResult]) -> list[MedicationResult]:
        seen: set[str] = set()
        unique: list[MedicationResult] = []
        for medication in medications:
            normalized = medication.name.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(medication)
        return unique

    def _dedupe_diagnoses(self, diagnoses: list[DiagnosisResult]) -> list[DiagnosisResult]:
        seen: set[str] = set()
        unique: list[DiagnosisResult] = []
        for diagnosis in diagnoses:
            normalized = diagnosis.term.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(diagnosis)
        return unique
