from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedMedication:
    canonical_name: str
    aliases: tuple[str, ...]


SEED_MEDICATIONS: tuple[SeedMedication, ...] = (
    SeedMedication("Lisinopril", ("Prinivil", "Zestril")),
    SeedMedication("Metformin", ("Glucophage",)),
    SeedMedication("Atorvastatin", ("Lipitor",)),
    SeedMedication("Levothyroxine", ("Synthroid", "Levoxyl")),
    SeedMedication("Amlodipine", ("Norvasc",)),
    SeedMedication("Omeprazole", ("Prilosec",)),
    SeedMedication("Losartan", ("Cozaar",)),
    SeedMedication("Albuterol", ("Ventolin", "ProAir", "Proventil")),
    SeedMedication("Hydrochlorothiazide", ("HCTZ", "Microzide")),
    SeedMedication("Sertraline", ("Zoloft",)),
)
