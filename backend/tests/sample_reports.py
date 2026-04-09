from __future__ import annotations

import fitz

LAB_REPORT_TEXT = """
Comprehensive Metabolic Panel
Glucose: 142 mg/dL (70-100)
Potassium: 3.1 mmol/L (3.5-5.1)
Sodium: 138 mmol/L (135-145)
"""

MEDICATION_REPORT_TEXT = """
Discharge Medications:
Lisinopril 10 mg daily
Metformin 500 mg twice daily
"""

DIAGNOSIS_REPORT_TEXT = """
Assessment:
Hypertension
Type 2 diabetes
"""

MIXED_REPORT_TEXT = """
Discharge Summary
Diagnosis: Hypertension; Hyperlipidemia
Glucose: 142 mg/dL (70-100)
Cholesterol: 240 mg/dL (0-200)
Discharge Medications: Lisinopril 10 mg daily, Atorvastatin 20 mg nightly
"""

UNKNOWN_REPORT_TEXT = """
Patient portal message:
Please call the office if symptoms continue.
"""


def build_pdf_bytes(text: str | None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(fitz.Rect(72, 72, 540, 720), text)
    return document.write()
