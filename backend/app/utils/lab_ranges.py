from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LabReference:
    low: float
    high: float
    unit: str
    description: str


DEFAULT_LAB_REFERENCES: dict[str, LabReference] = {
    "glucose": LabReference(70, 100, "mg/dL", "blood sugar level"),
    "potassium": LabReference(3.5, 5.1, "mmol/L", "a mineral that supports muscles and heart rhythm"),
    "sodium": LabReference(135, 145, "mmol/L", "a mineral that helps regulate fluid balance"),
    "creatinine": LabReference(0.6, 1.3, "mg/dL", "a marker often used to check kidney function"),
    "hemoglobin": LabReference(12.0, 17.0, "g/dL", "the protein in red blood cells that carries oxygen"),
    "wbc": LabReference(4.0, 11.0, "K/uL", "white blood cells that help fight infection"),
    "platelets": LabReference(150, 450, "K/uL", "cells that help blood clot"),
    "a1c": LabReference(4.0, 5.6, "%", "an average measure of blood sugar over the last few months"),
    "bun": LabReference(7, 20, "mg/dL", "a lab value often reviewed with kidney function"),
    "cholesterol": LabReference(0, 200, "mg/dL", "a fat-like substance measured in the blood"),
}

LAB_ALIASES = {
    "blood glucose": "glucose",
    "glucose fasting": "glucose",
    "k": "potassium",
    "na": "sodium",
    "serum creatinine": "creatinine",
    "hgb": "hemoglobin",
    "hemoglobin a1c": "a1c",
    "white blood cells": "wbc",
    "white blood cell count": "wbc",
    "plt": "platelets",
}


def normalize_lab_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", name.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return LAB_ALIASES.get(cleaned, cleaned)


def get_reference_for_lab(name: str) -> LabReference | None:
    return DEFAULT_LAB_REFERENCES.get(normalize_lab_name(name))


def parse_reference_range(range_text: str) -> tuple[float, float] | None:
    match = re.search(
        r"(?P<low>-?\d+(?:\.\d+)?)\s*(?:-|to)\s*(?P<high>-?\d+(?:\.\d+)?)",
        range_text,
        re.IGNORECASE,
    )
    if not match:
        return None
    return float(match.group("low")), float(match.group("high"))


def resolve_reference_range(name: str, range_text: str | None) -> tuple[float, float] | None:
    if range_text:
        parsed = parse_reference_range(range_text)
        if parsed:
            return parsed
    reference = get_reference_for_lab(name)
    if not reference:
        return None
    return reference.low, reference.high


def format_reference_range(name: str, range_text: str | None) -> str:
    if range_text:
        return range_text.strip()
    reference = get_reference_for_lab(name)
    if not reference:
        return ""
    return f"{reference.low:g}-{reference.high:g} {reference.unit}".strip()


def build_lab_explanation(name: str, status: str) -> str:
    reference = get_reference_for_lab(name)
    topic = reference.description if reference else "this lab value"
    if status == "low":
        return f"{name} is below the listed reference range. This can relate to changes in {topic}, but clinicians interpret it alongside symptoms and other results."
    if status == "high":
        return f"{name} is above the listed reference range. This may deserve follow-up, but the meaning depends on the rest of the report and your medical history."
    if status == "normal":
        return f"{name} is within the listed reference range, which usually means this measurement does not stand out on its own."
    return f"{name} was found in the report, but there was not enough range information to label it confidently."

