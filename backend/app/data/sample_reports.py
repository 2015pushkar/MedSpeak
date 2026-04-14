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

PAMELA_ROGERS_HP_TEXT = """
History and Physical Examination
Chief Complaint & ID: Ms. Rogers is a 56 y/o WF having chest pains for the last week.
She becomes short of breath during these episodes.
She was diagnosed with hypertension 3 years ago and had a TAH with BSO 6 years ago.
There is a family history of premature CAD.

Medical History â€“
1990: Diagnosed with peptic ulcer disease, which resolved after three months on cimetidine.

Allergy: Penicillin; experienced rash and hives in 1985.

Social History â€“
Medications: No prescription or illegal drug use.
Occasional OTC ibuprofen (Advil) for headache (QOD).

Musculoskeletal:
This pain is usually relieved with Tylenol.

Physical Examination
Vital Signs:
Blood Pressure 168/98, Pulse 90, Respirations 20, Temperature 37 degrees.

Surgical â€“
1998:
Total abdominal hysterectomy and bilateral oophorectomy for uterine fibroids.

Assessment and Differential Diagnosis
1. Chest pain with features of angina pectoris
2. Dyspnea
3. Recent onset hypertension and abdominal bruit
"""


def build_pdf_bytes(text: str | None) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(fitz.Rect(72, 72, 540, 720), text)
    return document.write()
