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
    AllergyResult,
    AnalysisMeta,
    AnalysisResponse,
    DiagnosisResult,
    DocumentType,
    LabResult,
    MedicationEvidence,
    MedicationResult,
    MedicationStatus,
    ProcessingTrace,
    RiskFactorResult,
    SurgeryResult,
    VitalResult,
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

KNOWN_PROBLEMS = {
    "chest pain": "Chest pain is discomfort in the chest area. It can have many causes, and a clinician should interpret it in context.",
    "shortness of breath": "Shortness of breath means breathing feels harder or less comfortable than usual.",
    "dyspnea": "Dyspnea means shortness of breath or difficulty breathing.",
    "hypertension": "High blood pressure. It means blood is pushing too strongly against the walls of your arteries.",
    "type 2 diabetes": "A condition where the body has trouble managing blood sugar over time.",
    "diabetes mellitus": "A condition where the body has trouble controlling blood sugar.",
    "hyperlipidemia": "Higher-than-desired fats such as cholesterol or triglycerides in the blood.",
    "anemia": "A lower-than-expected amount of healthy red blood cells or hemoglobin, which can reduce oxygen delivery.",
    "hypokalemia": "A lower-than-normal potassium level.",
    "pneumonia": "An infection or inflammation affecting the lungs.",
    "chronic kidney disease": "Long-term kidney damage or reduced kidney function.",
    "asthma": "A condition that can narrow the airways and make breathing harder.",
    "systolic murmur": "A murmur is an extra heart sound. It can reflect blood flow turbulence and may need follow-up testing.",
    "abdominal bruit": "An abdominal bruit is a sound heard over abdominal blood vessels and can suggest turbulent blood flow.",
    "epigastric pain": "Epigastric pain is discomfort in the upper middle part of the abdomen.",
    "low back pain": "Low back pain is discomfort in the lower back area.",
}

KNOWN_HISTORY_CONTEXT = {
    "peptic ulcer disease": "Past peptic ulcer disease is relevant history because it can affect how stomach symptoms and some medications are interpreted.",
    "family history of premature cad": "A family history of early coronary artery disease can increase concern about heart-related causes of symptoms.",
    "fh of early ascvd": "A family history of early atherosclerotic cardiovascular disease can increase heart risk.",
    "early surgical menopause": "Early loss of ovarian hormone production can change long-term cardiovascular risk.",
}

ACTIVE_PROBLEM_TERMS = (
    "chest pain",
    "shortness of breath",
    "dyspnea",
    "hypertension",
    "systolic murmur",
    "abdominal bruit",
    "epigastric pain",
    "low back pain",
)


@dataclass(frozen=True)
class CompletionResult:
    payload: dict[str, Any] | None
    failure_reason: str | None = None


@dataclass(frozen=True)
class MedicationExtractionBundle:
    medications: list[MedicationResult]
    source: str
    partial_reasons: list[str]
    fallback_used: bool


@dataclass(frozen=True)
class ClinicalContextBundle:
    diagnoses: list[DiagnosisResult]
    vitals: list[VitalResult]
    allergies: list[AllergyResult]
    surgeries: list[SurgeryResult]
    risk_factors: list[RiskFactorResult]
    source: str
    partial_reasons: list[str]
    fallback_used: bool


class MedicalPipeline:
    def __init__(self, settings: Settings, openfda_tool: OpenFDATool) -> None:
        self.settings = settings
        self.openfda_tool = openfda_tool
        self.client = self._build_client()
        self.medication_rag = MedicationRAGService.from_settings(settings, self.client)
        self.safety = SafetyService()
        self.last_debug: dict[str, Any] = {}
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
        self.last_debug = {}
        state: PipelineState = {
            "input_text": text,
            "warnings": [],
            "partial_data_reasons": partial_data_reasons or [],
            "fallback_used": False,
            "sources": [],
            "processing_trace": {},
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
        summary = self._align_summary_with_state(safety_result.summary, result)
        warnings = self._dedupe(safety_result.warnings)
        questions = self._prioritize_questions(result, self._dedupe(safety_result.questions_for_doctor)[:5])
        partial_reasons = self._dedupe([*partial_reasons, *safety_result.partial_data_reasons])

        trace = result.get("processing_trace", {})
        processing_trace = ProcessingTrace(
            classifier=trace.get("classifier", "heuristic"),
            medications=trace.get("medications", "heuristic"),
            diagnoses=trace.get("diagnoses", "heuristic"),
            synthesis=trace.get("synthesis", "template"),
        )

        return AnalysisResponse(
            document_type=result.get("document_type", "unknown"),
            summary=summary,
            warnings=warnings,
            labs=result.get("labs", []),
            medications=result.get("medications", []),
            diagnoses=result.get("diagnoses", []),
            vitals=result.get("vitals", []),
            allergies=result.get("allergies", []),
            surgeries=result.get("surgeries", []),
            risk_factors=result.get("risk_factors", []),
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
                processing_trace=processing_trace,
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
                "processing_trace": {"classifier": "llm"},
            }
        document_type, targets = self._heuristic_classification(state["input_text"])
        payload: PipelineState = {
            "document_type": document_type,
            "agent_targets": targets,
            "fallback_used": True,
            "processing_trace": {"classifier": "heuristic"},
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
        bundle = await self._extract_medications(state["input_text"])
        results: list[MedicationResult] = []
        partial_reasons = list(bundle.partial_reasons)
        sources: list[str] = []

        for medication in bundle.medications:
            resolved_name = self.medication_rag.resolve_name(medication.name)
            retrieval = await self.medication_rag.retrieve(medication.name, top_k=self.settings.retrieval_top_k)
            if retrieval.partial_reason:
                partial_reasons.append(retrieval.partial_reason)

            if resolved_name and retrieval.chunks:
                rag_grounding = await self.medication_rag.ground_medication(
                    medication.name,
                    medication.purpose or "This medication was mentioned in the document.",
                    top_k=self.settings.retrieval_top_k,
                )
                if rag_grounding:
                    results.append(
                        medication.model_copy(
                            update={
                                "purpose": rag_grounding.purpose,
                                "common_side_effects": rag_grounding.common_side_effects,
                                "cautions": rag_grounding.cautions,
                                "fda_enriched": True,
                                "grounding_status": "rag",
                                "grounding_note": "grounded from local corpus",
                                "evidence": rag_grounding.evidence,
                            }
                        )
                    )
                    sources.append(rag_grounding.backend)
                    continue

            try:
                enrichment = await self.openfda_tool.lookup(medication.name)
            except Exception:
                enrichment = MedicationEnrichment()
                partial_reasons.append(f"OpenFDA enrichment was unavailable for {medication.name}.")

            evidence = self._build_live_openfda_evidence(enrichment)
            if enrichment.fda_enriched:
                cached_for_future = await self.medication_rag.cache_openfda_document(medication.name, self.openfda_tool)
                grounded = medication.model_copy(
                    update={
                        "purpose": enrichment.purpose
                        or medication.purpose
                        or "This medication was mentioned in the document, but the report did not explain why it was used.",
                        "common_side_effects": enrichment.common_side_effects,
                        "cautions": enrichment.cautions,
                        "fda_enriched": True,
                        "grounding_status": "openfda_live",
                        "grounding_note": "enriched from live OpenFDA and saved for future local grounding"
                        if cached_for_future
                        else "enriched from live OpenFDA",
                        "evidence": evidence,
                    }
                )
                sources.append("openfda")
            else:
                grounded = medication.model_copy(
                    update={
                        "grounding_status": "text_only",
                        "grounding_note": "not in local corpus" if resolved_name is None else "mentioned only",
                        "evidence": [],
                    }
                )
            results.append(grounded)

        payload: PipelineState = {
            "medications": results,
            "processing_trace": {"medications": bundle.source},
        }
        if partial_reasons:
            payload["partial_data_reasons"] = self._dedupe(partial_reasons)
        if sources:
            payload["sources"] = self._dedupe(sources)
        if bundle.fallback_used:
            payload["fallback_used"] = True
        return payload

    async def _diagnosis_agent_node(self, state: PipelineState) -> PipelineState:
        bundle = await self._extract_clinical_context(state["input_text"])
        payload: PipelineState = {
            "diagnoses": bundle.diagnoses,
            "vitals": bundle.vitals,
            "allergies": bundle.allergies,
            "surgeries": bundle.surgeries,
            "risk_factors": bundle.risk_factors,
            "processing_trace": {"diagnoses": bundle.source},
        }
        if bundle.fallback_used:
            payload["fallback_used"] = True
        if bundle.partial_reasons:
            payload["partial_data_reasons"] = bundle.partial_reasons
        return payload

    async def _synthesis_node(self, state: PipelineState) -> PipelineState:
        warnings = self._build_warning_messages(state)
        questions = self._build_questions(state)
        summary = self._build_summary(state)

        if self.client:
            llm = await self._synthesize_with_llm(state, summary, warnings, questions)
            if llm.payload:
                normalized = self._normalize_synthesis_output(llm.payload, state)
                return {
                    "summary": normalized["summary"],
                    "warnings": normalized["warnings"],
                    "questions_for_doctor": normalized["questions_for_doctor"],
                    "processing_trace": {"synthesis": "llm"},
                }
            payload: PipelineState = {
                "summary": summary,
                "warnings": warnings,
                "questions_for_doctor": questions,
                "fallback_used": True,
                "processing_trace": {"synthesis": "template"},
            }
            if llm.failure_reason:
                payload["partial_data_reasons"] = [llm.failure_reason]
            return payload
        return {
            "summary": summary,
            "warnings": warnings,
            "questions_for_doctor": questions,
            "processing_trace": {"synthesis": "template"},
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

    async def _extract_medications(self, text: str) -> MedicationExtractionBundle:
        if self.client:
            completion = await self._json_completion(
                model=self.settings.openai_analyst_model,
                prompt=MEDICATION_PROMPT,
                user_content=text[:6000],
                failure_label="Medication extraction model",
            )
            if completion.payload:
                medications = [
                    self._build_medication_result(
                        text=text,
                        name=item["name"].strip(),
                        purpose=(item.get("purpose") or "").strip() or "This medication was mentioned in the document.",
                    )
                    for item in completion.payload.get("medications", [])
                    if item.get("name")
                ]
                if medications:
                    return MedicationExtractionBundle(
                        medications=self._dedupe_medications(medications),
                        source="llm",
                        partial_reasons=[],
                        fallback_used=False,
                    )
            if completion.failure_reason:
                return MedicationExtractionBundle(
                    medications=self._extract_medications_heuristically(text),
                    source="heuristic",
                    partial_reasons=[completion.failure_reason],
                    fallback_used=True,
                )
        return MedicationExtractionBundle(
            medications=self._extract_medications_heuristically(text),
            source="heuristic",
            partial_reasons=[],
            fallback_used=True,
        )

    async def _extract_clinical_context(self, text: str) -> ClinicalContextBundle:
        heuristic_bundle = self._extract_clinical_context_heuristically(text)
        if self.client:
            completion = await self._json_completion(
                model=self.settings.openai_analyst_model,
                prompt=DIAGNOSIS_PROMPT,
                user_content=text[:6000],
                failure_label="Diagnosis explanation model",
            )
            if completion.payload:
                llm_bundle = self._build_clinical_context_from_payload(completion.payload)
                return self._merge_context_bundles(
                    self._reclassify_context_bundle(text, llm_bundle),
                    heuristic_bundle,
                    source="llm",
                    partial_reasons=[],
                    fallback_used=False,
                )
            if completion.failure_reason:
                return ClinicalContextBundle(
                    diagnoses=heuristic_bundle.diagnoses,
                    vitals=heuristic_bundle.vitals,
                    allergies=heuristic_bundle.allergies,
                    surgeries=heuristic_bundle.surgeries,
                    risk_factors=heuristic_bundle.risk_factors,
                    source="heuristic",
                    partial_reasons=[completion.failure_reason],
                    fallback_used=True,
                )
        return heuristic_bundle

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
                    "vitals": [item.model_dump() for item in state.get("vitals", [])],
                    "allergies": [item.model_dump() for item in state.get("allergies", [])],
                    "surgeries": [item.model_dump() for item in state.get("surgeries", [])],
                    "risk_factors": [item.model_dump() for item in state.get("risk_factors", [])],
                }
            ),
            failure_label="Synthesis model",
        )
        self.last_debug["synthesizer_response"] = {
            "payload": completion.payload,
            "failure_reason": completion.failure_reason,
            "used_llm": bool(completion.payload),
        }
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
        lab_score = len(
            re.findall(r"\b(glucose|potassium|sodium|hemoglobin|a1c|creatinine|wbc|platelets|bun)\b", text, re.IGNORECASE)
        )
        lab_score += len(re.findall(r"\d+(?:\.\d+)?\s*(?:mg/dl|mmol/l|g/dl|k/ul|%)", text, re.IGNORECASE))

        medication_mentions = self._extract_medications_heuristically(text)
        med_score = len(medication_mentions) * 2
        med_score += len(re.findall(r"\b(?:medications?|tablet|capsule|daily|bid|tid|prn|otc)\b", text, re.IGNORECASE))

        diagnosis_score = len(
            re.findall(r"\b(?:diagnosis|impression|assessment|history of|discharge diagnosis|problem list|differential diagnosis)\b", text, re.IGNORECASE)
        )
        diagnosis_score += len(self._extract_active_problem_matches(text))

        hnp_markers = len(
            re.findall(
                r"\b(?:history and physical|past medical history|review of systems|physical examination|vital signs|family history)\b",
                text,
                re.IGNORECASE,
            )
        )

        targets: list[str] = []
        if lab_score >= 2:
            targets.append("lab_agent")
        if med_score >= 2:
            targets.append("medication_agent")
        if diagnosis_score >= 1 or hnp_markers >= 2:
            targets.append("diagnosis_agent")
        if hnp_markers >= 2 and targets == ["diagnosis_agent"] and med_score >= 1:
            targets.append("medication_agent")

        targets = self._dedupe(targets)
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
        section_pattern = re.compile(r"(?:medications?|discharge medications?|current medications?)\s*[:\-]\s*(?P<items>.+)", re.IGNORECASE)
        narrative_patterns = (
            re.compile(r"\bresolved after [^.]{0,80}\bon\s+(?P<name>[A-Za-z][A-Za-z]+(?:\s*\([^)]+\))?)", re.IGNORECASE),
            re.compile(r"\bOTC\s+(?P<name>[A-Za-z][A-Za-z]+(?:\s*\([^)]+\))?)", re.IGNORECASE),
            re.compile(r"\brelieved with\s+(?P<name>[A-Za-z][A-Za-z]+(?:\s*\([^)]+\))?)", re.IGNORECASE),
        )

        for line in text.splitlines():
            section_match = section_pattern.search(line)
            if section_match:
                items = re.split(r"[,;]", section_match.group("items"))
                for item in items:
                    match = re.search(r"([A-Za-z][A-Za-z]+(?:\s*\([^)]+\))?)", item.strip())
                    if match:
                        candidates.append(match.group(1))
            line_match = line_pattern.search(line)
            if line_match:
                candidates.append(line_match.group("name"))
            for pattern in narrative_patterns:
                narrative_match = pattern.search(line)
                if narrative_match:
                    candidates.append(narrative_match.group("name"))

        medications = [
            self._build_medication_result(
                text=text,
                name=re.sub(r"\s+", " ", candidate).strip().title(),
                purpose="This medication was mentioned in the document.",
            )
            for candidate in candidates
            if candidate
        ]
        return self._dedupe_medications(medications)

    def _extract_clinical_context_heuristically(self, text: str) -> ClinicalContextBundle:
        active_diagnoses = [
            DiagnosisResult(term=term, plain_language=plain_language)
            for term, plain_language in self._extract_active_problem_matches(text).items()
        ]
        return ClinicalContextBundle(
            diagnoses=self._dedupe_diagnoses(active_diagnoses),
            vitals=self._dedupe_vitals(self._extract_vitals_heuristically(text)),
            allergies=self._dedupe_allergies(self._extract_allergies_heuristically(text)),
            surgeries=self._dedupe_surgeries(self._extract_surgeries_heuristically(text)),
            risk_factors=self._dedupe_risk_factors(self._extract_risk_factors_heuristically(text)),
            source="heuristic",
            partial_reasons=[],
            fallback_used=True,
        )

    def _build_clinical_context_from_payload(self, payload: dict[str, Any]) -> ClinicalContextBundle:
        diagnoses = [
            DiagnosisResult(
                term=item["term"].strip(),
                plain_language=item.get("plain_language", "").strip()
                or "This is a clinical problem noted in the report. Ask your clinician how it applies in your case.",
            )
            for item in payload.get("diagnoses", [])
            if item.get("term")
        ]
        allergies = [
            AllergyResult(
                substance=str(item.get("substance", "")).strip(),
                reaction=str(item.get("reaction", "")).strip(),
            )
            for item in payload.get("allergies", [])
            if item.get("substance")
        ]
        surgeries = [
            SurgeryResult(
                procedure=str(item.get("procedure", "")).strip(),
                timing=str(item.get("timing", "")).strip(),
                reason=str(item.get("reason", "")).strip(),
            )
            for item in payload.get("surgeries", [])
            if item.get("procedure")
        ]
        risk_factors = [
            RiskFactorResult(
                factor=str(item.get("factor", "")).strip(),
                plain_language=str(item.get("plain_language", "")).strip()
                or "This is background medical context that can matter when a clinician interprets the rest of the report.",
            )
            for item in payload.get("risk_factors", [])
            if item.get("factor")
        ]
        return ClinicalContextBundle(
            diagnoses=diagnoses,
            vitals=[],
            allergies=allergies,
            surgeries=surgeries,
            risk_factors=risk_factors,
            source="llm",
            partial_reasons=[],
            fallback_used=False,
        )

    def _reclassify_context_bundle(self, text: str, bundle: ClinicalContextBundle) -> ClinicalContextBundle:
        diagnoses: list[DiagnosisResult] = []
        allergies = list(bundle.allergies)
        surgeries = list(bundle.surgeries)
        risk_factors = list(bundle.risk_factors)

        for diagnosis in bundle.diagnoses:
            normalized = diagnosis.term.lower().strip()
            if self._looks_like_allergy(normalized):
                allergies.append(
                    AllergyResult(
                        substance=re.sub(r"(?i)\ballergy to\b", "", diagnosis.term).strip(),
                        reaction="",
                    )
                )
            elif self._looks_like_surgery(normalized):
                surgeries.append(SurgeryResult(procedure=diagnosis.term))
            elif self._looks_like_risk_or_history(normalized, text):
                risk_factors.append(RiskFactorResult(factor=diagnosis.term, plain_language=diagnosis.plain_language))
            else:
                diagnoses.append(diagnosis)

        return ClinicalContextBundle(
            diagnoses=self._dedupe_diagnoses(diagnoses),
            vitals=self._dedupe_vitals(bundle.vitals),
            allergies=self._dedupe_allergies(allergies),
            surgeries=self._dedupe_surgeries(surgeries),
            risk_factors=self._dedupe_risk_factors(risk_factors),
            source=bundle.source,
            partial_reasons=bundle.partial_reasons,
            fallback_used=bundle.fallback_used,
        )

    def _merge_context_bundles(
        self,
        primary: ClinicalContextBundle,
        supplement: ClinicalContextBundle,
        *,
        source: str,
        partial_reasons: list[str],
        fallback_used: bool,
    ) -> ClinicalContextBundle:
        return ClinicalContextBundle(
            diagnoses=self._dedupe_diagnoses([*primary.diagnoses, *supplement.diagnoses]),
            vitals=self._dedupe_vitals([*primary.vitals, *supplement.vitals]),
            allergies=self._dedupe_allergies([*primary.allergies, *supplement.allergies]),
            surgeries=self._dedupe_surgeries([*primary.surgeries, *supplement.surgeries]),
            risk_factors=self._dedupe_risk_factors([*primary.risk_factors, *supplement.risk_factors]),
            source=source,
            partial_reasons=partial_reasons,
            fallback_used=fallback_used,
        )

    def _extract_active_problem_matches(self, text: str) -> dict[str, str]:
        matches: dict[str, str] = {}
        lowered_text = text.lower()
        synonyms = {
            "shortness of breath": ("shortness of breath", "dyspnea"),
        }

        for term in ACTIVE_PROBLEM_TERMS:
            candidates = synonyms.get(term, (term,))
            if any(candidate in lowered_text for candidate in candidates):
                display_term = "Shortness of breath" if term == "shortness of breath" else term.title()
                matches[display_term] = KNOWN_PROBLEMS[term]
        return matches

    def _extract_vitals_heuristically(self, text: str) -> list[VitalResult]:
        vitals: list[VitalResult] = []
        patterns = {
            "Blood Pressure": (r"blood pressure\s*:?\s*(\d{2,3}/\d{2,3})", ""),
            "Pulse": (r"pulse\s*:?\s*(\d{2,3})", "bpm"),
            "Respirations": (r"respirations?\s*:?\s*(\d{1,2})", "breaths/min"),
            "Temperature": (r"temperature\s*:?\s*(\d+(?:\.\d+)?)\s*(degrees|f|c)?", ""),
        }
        lowered = text.lower()
        for name, (pattern, default_unit) in patterns.items():
            match = re.search(pattern, lowered, re.IGNORECASE)
            if not match:
                continue
            unit = default_unit
            if name == "Temperature" and match.lastindex and match.group(2):
                unit = match.group(2)
            vitals.append(VitalResult(name=name, value=match.group(1), unit=unit))
        return vitals

    def _extract_allergies_heuristically(self, text: str) -> list[AllergyResult]:
        allergies: list[AllergyResult] = []
        for match in re.finditer(
            r"allerg(?:y|ies)\s*:\s*(?P<substance>[^;\n.]+)(?:;\s*(?:experienced|reaction[: ]+)?(?P<reaction>[^.\n]+))?",
            text,
            re.IGNORECASE,
        ):
            allergies.append(
                AllergyResult(
                    substance=match.group("substance").strip(),
                    reaction=(match.group("reaction") or "").strip(),
                )
            )
        return allergies

    def _extract_surgeries_heuristically(self, text: str) -> list[SurgeryResult]:
        surgeries: list[SurgeryResult] = []
        year_context = ""
        for raw_line in text.splitlines():
            line = re.sub(r"\s+", " ", raw_line).strip()
            if not line:
                continue
            year_match = re.fullmatch(r"(\d{4}):", line)
            if year_match:
                year_context = year_match.group(1)
                continue
            if re.search(r"\b(?:hysterectomy|oophorectomy|bunionectomy|appendectomy|cholecystectomy|mastectomy|arthroplasty|bypass)\b", line, re.IGNORECASE):
                reason_match = re.search(r"\bfor\b\s+(.+)$", line, re.IGNORECASE)
                surgeries.append(
                    SurgeryResult(
                        procedure=re.sub(r"\bfor\b.+$", "", line, flags=re.IGNORECASE).strip(" ."),
                        timing=year_context,
                        reason=(reason_match.group(1).strip(" .") if reason_match else ""),
                    )
                )
                year_context = ""
        shorthand_match = re.search(r"\bTAH with BSO (\d+) years ago\b", text, re.IGNORECASE)
        if shorthand_match:
            surgeries.append(
                SurgeryResult(
                    procedure="TAH with BSO",
                    timing=f"{shorthand_match.group(1)} years ago",
                )
            )
        return surgeries

    def _extract_risk_factors_heuristically(self, text: str) -> list[RiskFactorResult]:
        factors: list[RiskFactorResult] = []
        lowered = text.lower()
        if "family history of premature cad" in lowered:
            factors.append(
                RiskFactorResult(
                    factor="Family history of premature CAD",
                    plain_language=KNOWN_HISTORY_CONTEXT["family history of premature cad"],
                )
            )
        elif "fh of early ascvd" in lowered:
            factors.append(
                RiskFactorResult(
                    factor="FH of early ASCVD",
                    plain_language=KNOWN_HISTORY_CONTEXT["fh of early ascvd"],
                )
            )
        if re.search(r"history of peptic ulcer disease", lowered) or re.search(r"peptic ulcer disease, which\s+resolved", lowered):
            factors.append(
                RiskFactorResult(
                    factor="History of peptic ulcer disease",
                    plain_language=KNOWN_HISTORY_CONTEXT["peptic ulcer disease"],
                )
            )
        return factors

    def _build_medication_result(self, *, text: str, name: str, purpose: str) -> MedicationResult:
        status = self._infer_medication_status(text, name)
        return MedicationResult(
            name=name,
            purpose=purpose,
            common_side_effects=[],
            cautions=[],
            fda_enriched=False,
            grounding_status="text_only",
            status=status,
            grounding_note="mentioned only",
            evidence=[],
        )

    def _infer_medication_status(self, text: str, medication_name: str) -> MedicationStatus:
        contexts = self._medication_context_windows(text, medication_name)
        joined = " ".join(contexts).lower()

        historical_markers = (
            "resolved after",
            "history of",
            "past medical history",
            "stopped after",
            "because of drowsiness",
            "years ago",
            "medical history",
            "was on",
        )
        otc_markers = (
            "otc",
            "occasional",
            "relieved with",
            "for headache",
            "for pain",
            "as needed",
            "prn",
            "qod",
        )
        current_markers = (
            "discharge medications",
            "current medications",
            "home medications",
            "take ",
            "daily",
            "twice daily",
            "nightly",
        )

        if any(marker in joined for marker in historical_markers):
            return "historical"
        if any(marker in joined for marker in otc_markers):
            return "otc_prn"
        if any(marker in joined for marker in current_markers):
            return "current"
        return "unclear"

    def _medication_context_windows(self, text: str, medication_name: str) -> list[str]:
        lines = text.splitlines()
        candidates = self._medication_name_candidates(medication_name)
        contexts: list[str] = []
        for index, line in enumerate(lines):
            normalized = re.sub(r"\s+", " ", line).strip().lower()
            if not normalized:
                continue
            if any(candidate in normalized for candidate in candidates):
                window = lines[max(index - 1, 0) : min(index + 2, len(lines))]
                contexts.append(" ".join(part.strip() for part in window if part.strip()))
        return contexts or [text[:400]]

    def _medication_name_candidates(self, medication_name: str) -> list[str]:
        raw = medication_name.lower().strip()
        candidates = [raw]
        parenthetical = re.search(r"^(?P<outside>.+?)\s*\((?P<inside>[^)]+)\)$", raw)
        if parenthetical:
            outside = parenthetical.group("outside").strip()
            inside = parenthetical.group("inside").strip()
            if outside:
                candidates.append(outside)
            if inside:
                candidates.append(inside)
        return list(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _looks_like_allergy(self, normalized_term: str) -> bool:
        return "allergy" in normalized_term or "allergic" in normalized_term

    def _looks_like_surgery(self, normalized_term: str) -> bool:
        return bool(re.search(r"\b(?:hysterectomy|oophorectomy|tah|bso|ectomy|surgery|operative)\b", normalized_term))

    def _looks_like_risk_or_history(self, normalized_term: str, text: str) -> bool:
        if "family history" in normalized_term or normalized_term.startswith("fh of"):
            return True
        if normalized_term in {"peptic ulcer disease", "history of peptic ulcer disease"} and re.search(
            r"history of peptic ulcer disease|resolved after three months on cimetidine",
            text,
            re.IGNORECASE,
        ):
            return True
        return False

    def _normalize_synthesis_output(self, payload: dict[str, Any], state: PipelineState) -> dict[str, list[str] | str]:
        summary = str(payload.get("summary", "")).strip() or self._build_summary(state)
        warnings = [str(item).strip() for item in payload.get("warnings", []) if str(item).strip()] or self._build_warning_messages(state)
        questions = [str(item).strip() for item in payload.get("questions_for_doctor", []) if str(item).strip()] or self._build_questions(state)
        return {
            "summary": self._align_summary_with_state(summary, state),
            "warnings": self._dedupe(warnings),
            "questions_for_doctor": self._prioritize_questions(state, questions),
        }

    def _build_summary(self, state: PipelineState) -> str:
        labs = state.get("labs", [])
        medications = state.get("medications", [])
        diagnoses = state.get("diagnoses", [])
        vitals = state.get("vitals", [])
        allergies = state.get("allergies", [])
        surgeries = state.get("surgeries", [])
        risk_factors = state.get("risk_factors", [])
        document_type = state.get("document_type", "unknown")

        if document_type == "unknown" and not any([labs, medications, diagnoses, vitals, allergies, surgeries, risk_factors]):
            return "MedSpeak could not confidently identify a familiar report pattern from the text that was provided."

        fragments: list[str] = []
        if diagnoses:
            problem_names = ", ".join(item.term.lower() for item in diagnoses[:3])
            fragments.append(f"It focuses on active problems including {problem_names}.")
        if vitals:
            vital_summary = self._summarize_vitals(vitals)
            if vital_summary:
                fragments.append(vital_summary)
        if labs:
            abnormal = [lab.name for lab in labs if lab.status in {"low", "high"}]
            if abnormal:
                fragments.append(f"It includes {len(labs)} lab results, with out-of-range findings such as {', '.join(abnormal[:2])}.")
            else:
                fragments.append(f"It includes {len(labs)} lab results without obvious out-of-range values.")
        current_meds = [med for med in medications if med.status == "current"]
        historical_meds = [med for med in medications if med.status == "historical"]
        otc_meds = [med for med in medications if med.status == "otc_prn"]
        unclear_meds = [med for med in medications if med.status == "unclear"]
        if current_meds:
            fragments.append(f"It lists {len(current_meds)} current medication{'s' if len(current_meds) != 1 else ''}.")
        if historical_meds:
            fragments.append(f"It also mentions {len(historical_meds)} historical medication treatment{'s' if len(historical_meds) != 1 else ''}.")
        if otc_meds:
            fragments.append(f"It notes {len(otc_meds)} over-the-counter or as-needed medication mention{'s' if len(otc_meds) != 1 else ''}.")
        if not current_meds and not historical_meds and not otc_meds and unclear_meds:
            fragments.append(f"It mentions {len(unclear_meds)} medication name{'s' if len(unclear_meds) != 1 else ''} without a clear active medication list.")
        if allergies:
            fragments.append(f"It documents {len(allergies)} allerg{'y' if len(allergies) == 1 else 'ies'}, including {allergies[0].substance}.")
        if surgeries:
            fragments.append("It includes prior surgical history.")
        if risk_factors:
            fragments.append("It also notes important family-history or past-history context.")
        return " ".join(fragments) or "This document was processed and organized into plain-language findings."

    def _summarize_vitals(self, vitals: list[VitalResult]) -> str:
        blood_pressure = next((item for item in vitals if item.name == "Blood Pressure"), None)
        if blood_pressure:
            match = re.match(r"(?P<systolic>\d{2,3})/(?P<diastolic>\d{2,3})", blood_pressure.value)
            if match and (int(match.group("systolic")) >= 140 or int(match.group("diastolic")) >= 90):
                return f"The recorded blood pressure is {blood_pressure.value}, which is elevated."
            return f"The document includes vital signs such as blood pressure {blood_pressure.value}."
        if vitals:
            return "The document includes recorded vital signs."
        return ""

    def _build_warning_messages(self, state: PipelineState) -> list[str]:
        warnings = list(state.get("warnings", []))
        labs = state.get("labs", [])
        medications = state.get("medications", [])
        vitals = state.get("vitals", [])

        if any(lab.status in {"low", "high"} for lab in labs):
            warnings.append("Some lab values appear outside the listed reference range.")
        if self._has_elevated_blood_pressure(vitals):
            warnings.append("The recorded blood pressure is elevated in this document.")
        if any(medication.cautions for medication in medications if medication.status == "current"):
            warnings.append("Medication label cautions were found for current medications. Ask a clinician how they apply to you.")
        if any(medication.grounding_status == "text_only" for medication in medications):
            warnings.append("Some medication explanations were text-only because grounded label context was unavailable.")
        if state.get("document_type") == "unknown":
            warnings.append("This text did not clearly match a standard lab, medication, or diagnosis report format.")
        warnings.extend(state.get("partial_data_reasons", []))
        return self._dedupe(warnings)

    def _build_questions(self, state: PipelineState) -> list[str]:
        questions: list[str] = []
        diagnoses = {self._canonicalize_problem_term(item.term): item for item in state.get("diagnoses", [])}
        vitals = state.get("vitals", [])

        if "chest pain" in diagnoses:
            questions.append("What is the most important next step to evaluate my chest pain and the risk of a heart-related cause?")
        if "shortness of breath" in diagnoses or "dyspnea" in diagnoses:
            questions.append("How should my shortness of breath be evaluated in relation to the chest pain and exam findings?")
        if "hypertension" in diagnoses or self._has_elevated_blood_pressure(vitals):
            questions.append("What does the elevated blood pressure in this report mean, and does it change the follow-up plan?")

        for lab in state.get("labs", []):
            if lab.status == "high":
                questions.append(f"What might explain my high {lab.name.lower()} result, and does it need follow-up?")
            elif lab.status == "low":
                questions.append(f"What might explain my low {lab.name.lower()} result, and should it be rechecked?")

        for medication in state.get("medications", []):
            if medication.status == "current" and (medication.cautions or medication.grounding_status != "text_only"):
                questions.append(f"How does {medication.name} fit with the rest of this report, and do I need any monitoring?")

        if not questions:
            questions.append("What are the key takeaways from this report, and is any follow-up needed?")
        return self._dedupe(questions)

    def _prioritize_questions(self, state: PipelineState, questions: list[str]) -> list[str]:
        blocked_medications = [
            medication.name.lower()
            for medication in state.get("medications", [])
            if medication.status in {"historical", "otc_prn", "unclear"}
        ]
        filtered = [
            question
            for question in questions
            if not any(name in question.lower() for name in blocked_medications)
        ]
        prioritized = [*self._build_questions(state), *filtered]
        return self._dedupe(prioritized)[:5]

    def _align_summary_with_state(self, summary: str, state: PipelineState) -> str:
        medications = state.get("medications", [])
        current_meds = [med for med in medications if med.status == "current"]
        if medications and not current_meds and re.search(r"\b(?:taking|takes|is taking|taking medications?|medications? like)\b", summary, re.IGNORECASE):
            return self._build_summary(state)
        return summary

    def _has_elevated_blood_pressure(self, vitals: list[VitalResult]) -> bool:
        blood_pressure = next((item for item in vitals if item.name == "Blood Pressure"), None)
        if not blood_pressure:
            return False
        match = re.match(r"(?P<systolic>\d{2,3})/(?P<diastolic>\d{2,3})", blood_pressure.value)
        if not match:
            return False
        return int(match.group("systolic")) >= 140 or int(match.group("diastolic")) >= 90

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
            normalized = re.sub(r"\([^)]*\)", "", medication.name.lower()).strip()
            if normalized in seen or normalized in {"glucose", "potassium", "sodium"}:
                continue
            seen.add(normalized)
            unique.append(medication)
        return unique

    def _dedupe_diagnoses(self, diagnoses: list[DiagnosisResult]) -> list[DiagnosisResult]:
        seen: set[str] = set()
        unique: list[DiagnosisResult] = []
        for diagnosis in diagnoses:
            normalized = self._canonicalize_problem_term(diagnosis.term)
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(diagnosis)
        return unique

    def _dedupe_vitals(self, vitals: list[VitalResult]) -> list[VitalResult]:
        seen: set[str] = set()
        unique: list[VitalResult] = []
        for vital in vitals:
            normalized = vital.name.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            unique.append(vital)
        return unique

    def _dedupe_allergies(self, allergies: list[AllergyResult]) -> list[AllergyResult]:
        seen: set[str] = set()
        unique: list[AllergyResult] = []
        for allergy in allergies:
            normalized = allergy.substance.lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(allergy)
        return unique

    def _dedupe_surgeries(self, surgeries: list[SurgeryResult]) -> list[SurgeryResult]:
        seen: set[str] = set()
        unique: list[SurgeryResult] = []
        for surgery in surgeries:
            normalized = self._canonicalize_surgery_term(surgery.procedure)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(surgery)
        return unique

    def _dedupe_risk_factors(self, risk_factors: list[RiskFactorResult]) -> list[RiskFactorResult]:
        seen: set[str] = set()
        unique: list[RiskFactorResult] = []
        for factor in risk_factors:
            normalized = factor.factor.lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique.append(factor)
        return unique

    def _canonicalize_problem_term(self, value: str) -> str:
        normalized = value.lower().strip()
        if "chest pain" in normalized:
            return "chest pain"
        if "dyspnea" in normalized or "shortness of breath" in normalized:
            return "shortness of breath"
        if "hypertension" in normalized:
            return "hypertension"
        if "abdominal bruit" in normalized:
            return "abdominal bruit"
        if "murmur" in normalized:
            return "systolic murmur"
        return normalized

    def _canonicalize_surgery_term(self, value: str) -> str:
        normalized = value.lower().strip()
        if re.search(r"\b(?:tah|hysterectomy)\b", normalized) and re.search(r"\b(?:bso|oophorectomy)\b", normalized):
            return "hysterectomy-oophorectomy"
        return normalized
