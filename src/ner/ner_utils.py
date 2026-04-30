from __future__ import annotations

import json
from pathlib import Path

DRUG_DB_PATH = Path("data/processed/drug_interactions.json")

# Severity order for sorting
SEVERITY_ORDER = {"major": 0, "moderate": 1, "minor": 2}


def load_drug_db(path: str | Path = DRUG_DB_PATH) -> dict:
    """Load drug interaction database from JSON."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def check_drug_interactions(medications: list[str], db: dict) -> list[dict]:
    """
    Check a list of medication names against the local drug interaction DB.

    DB format: {
        "drug_a::drug_b": {
            "severity": "major|moderate|minor",
            "description": "...",
            "recommendation": "..."
        }
    }

    Returns list of interaction dicts sorted by severity.
    """
    warnings = []
    seen = set()
    meds_lower = [m.lower().strip() for m in medications]

    for i, med_a in enumerate(meds_lower):
        for med_b in meds_lower[i + 1:]:
            key1 = f"{med_a}::{med_b}"
            key2 = f"{med_b}::{med_a}"
            pair_key = tuple(sorted([med_a, med_b]))
            if pair_key in seen:
                continue
            seen.add(pair_key)

            interaction = db.get(key1) or db.get(key2)
            if interaction:
                warnings.append({
                    "drug_a": med_a,
                    "drug_b": med_b,
                    "severity": interaction.get("severity", "unknown"),
                    "description": interaction.get("description", ""),
                    "recommendation": interaction.get("recommendation", ""),
                })

    warnings.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 99))
    return warnings


def normalize_entity_text(text: str) -> str:
    """Lowercase and strip punctuation for entity normalization."""
    import re
    return re.sub(r"[^\w\s-]", "", text.lower()).strip()


def deduplicate_entities(entities: list[str]) -> list[str]:
    seen = set()
    out = []
    for e in entities:
        norm = normalize_entity_text(e)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(e)
    return out


def build_sample_drug_db(output_path: str | Path = DRUG_DB_PATH) -> None:
    """Write a small sample drug interaction DB for testing."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample = {
        "warfarin::aspirin": {
            "severity": "major",
            "description": "Concurrent use increases bleeding risk significantly.",
            "recommendation": "Avoid combination; if necessary, monitor INR closely.",
        },
        "metformin::alcohol": {
            "severity": "moderate",
            "description": "Alcohol potentiates lactic acidosis risk with metformin.",
            "recommendation": "Advise patient to limit alcohol intake.",
        },
        "lisinopril::potassium": {
            "severity": "moderate",
            "description": "ACE inhibitors can increase serum potassium levels.",
            "recommendation": "Monitor serum potassium; avoid potassium supplements.",
        },
        "simvastatin::amiodarone": {
            "severity": "major",
            "description": "Risk of myopathy and rhabdomyolysis increased.",
            "recommendation": "Limit simvastatin dose; consider alternative statin.",
        },
    }
    with open(output_path, "w") as f:
        json.dump(sample, f, indent=2)
