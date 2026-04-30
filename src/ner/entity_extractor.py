from __future__ import annotations

import json
import re
from typing import Optional

import spacy
from spacy.language import Language

from src.ner.ner_utils import deduplicate_entities, check_drug_interactions, load_drug_db

PREFERRED_MODELS = ["en_core_sci_lg", "en_core_sci_sm", "en_core_web_sm"]

ENTITY_LABEL_MAP = {
    "DISEASE": "diagnoses",
    "CHEMICAL": "medications",
    "SYMPTOM": "symptoms",
    "DOSAGE": "dosages",
    "ANATOMY": "anatomy",
    "PROCEDURE": "procedures",
    "ORG": "procedures",
}

# ── Comprehensive clinical vocabulary for rule-based matching ─────────────────

SYMPTOMS = [
    "cough", "fever", "dyspnea", "shortness of breath", "chest pain",
    "fatigue", "hemoptysis", "wheezing", "edema", "nausea", "vomiting",
    "diarrhea", "headache", "dizziness", "syncope", "tachycardia",
    "bradycardia", "palpitations", "orthopnea", "night sweats",
    "weight loss", "anorexia", "malaise", "rigors", "chills",
    "pleuritic chest pain", "productive cough", "dry cough",
    "ankle swelling", "leg swelling", "abdominal pain", "haemoptysis",
    "dysphagia", "hoarseness", "stridor", "cyanosis", "jaundice",
]

DIAGNOSES = [
    "pneumonia", "community-acquired pneumonia", "cap",
    "heart failure", "heart failure with reduced ejection fraction", "hfref",
    "copd", "chronic obstructive pulmonary disease",
    "tuberculosis", "tb", "pulmonary tuberculosis",
    "pleural effusion", "malignant pleural effusion",
    "pneumothorax", "tension pneumothorax",
    "pulmonary embolism", "pe",
    "atelectasis", "consolidation", "interstitial pneumonia",
    "viral pneumonia", "aspiration pneumonia", "mycoplasma",
    "cardiomegaly", "pulmonary edema",
    "sepsis", "bacteraemia", "bacteremia",
    "lung adenocarcinoma", "lung cancer", "malignancy",
    "hypertension", "diabetes", "type 2 diabetes",
    "hyperlipidemia", "hyperlipidaemia",
    "emphysema", "fibrosis", "pulmonary fibrosis",
    "effusion", "hernia", "nodule", "mass",
]

MEDICATIONS = [
    "amoxicillin", "amoxicillin-clavulanate", "amoxicillin clavulanate",
    "co-amoxiclav", "azithromycin", "clarithromycin", "doxycycline",
    "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "ceftriaxone", "cefuroxime", "penicillin", "piperacillin",
    "tazobactam", "metronidazole", "vancomycin",
    "prednisolone", "prednisone", "dexamethasone", "hydrocortisone",
    "salbutamol", "albuterol", "ipratropium", "tiotropium",
    "furosemide", "frusemide", "spironolactone",
    "lisinopril", "ramipril", "enalapril", "perindopril",
    "carvedilol", "bisoprolol", "metoprolol", "atenolol",
    "sacubitril", "sacubitril-valsartan", "valsartan", "losartan",
    "amlodipine", "nifedipine", "diltiazem",
    "warfarin", "apixaban", "rivaroxaban", "dabigatran", "heparin",
    "aspirin", "clopidogrel", "ticagrelor",
    "atorvastatin", "simvastatin", "rosuvastatin", "pravastatin",
    "metformin", "insulin", "sitagliptin", "empagliflozin",
    "rifampicin", "isoniazid", "pyrazinamide", "ethambutol",
    "ibuprofen", "naproxen", "diclofenac", "paracetamol", "codeine",
    "morphine", "tramadol", "oxycodone",
    "omeprazole", "pantoprazole", "lansoprazole",
    "ondansetron", "metoclopramide",
    "oxygen", "supplemental oxygen",
]

DOSAGES_UNITS = ["mg", "mcg", "g", "ml", "units", "iu", "meq", "mmol", "mmhg"]

ANATOMY = [
    "lung", "lungs", "heart", "liver", "kidney", "kidneys",
    "spleen", "brain", "pleura", "diaphragm", "aorta",
    "trachea", "bronchi", "bronchus", "alveoli",
    "right lower lobe", "left lower lobe",
    "right upper lobe", "left upper lobe",
    "right middle lobe", "right base", "left base",
    "bilateral", "left lung", "right lung",
    "mediastinum", "hilum", "carina", "chest",
    "upper lobe", "lower lobe", "pericardium",
]

PROCEDURES = [
    "chest x-ray", "cxr", "ct scan", "ct chest", "hrct",
    "echocardiogram", "echo", "spirometry",
    "thoracentesis", "bronchoscopy", "biopsy",
    "blood culture", "sputum culture", "afb smear",
    "geneXpert", "pcr", "d-dimer", "bnp", "troponin",
    "full blood count", "fbc", "chest tube",
]


class ClinicalNERExtractor:
    """Rule-based clinical NER with comprehensive medical vocabulary."""

    def __init__(self):
        self._nlp: Optional[Language] = None
        self._drug_db: dict = {}

    def load(self) -> None:
        if self._nlp is not None:
            return

        for model_name in PREFERRED_MODELS:
            try:
                self._nlp = spacy.load(model_name)
                break
            except OSError:
                continue

        if self._nlp is None:
            raise RuntimeError(
                "No spaCy model found.\n"
                "Run: venv/Scripts/python.exe -m spacy download en_core_web_sm"
            )

        self._add_rule_patterns()
        self._drug_db = load_drug_db()

    def _add_rule_patterns(self) -> None:
        if "entity_ruler" in self._nlp.pipe_names:
            return

        # Add before NER so rules take priority
        ruler = self._nlp.add_pipe(
            "entity_ruler",
            before="ner" if "ner" in self._nlp.pipe_names else None,
            config={"overwrite_ents": True},
        )

        patterns = []

        # Multi-word and single-word patterns — all lowercased
        for term in SYMPTOMS:
            patterns.append({"label": "SYMPTOM", "pattern": term})

        for term in DIAGNOSES:
            patterns.append({"label": "DISEASE", "pattern": term})

        for term in MEDICATIONS:
            patterns.append({"label": "CHEMICAL", "pattern": term})

        for term in ANATOMY:
            patterns.append({"label": "ANATOMY", "pattern": term})

        for term in PROCEDURES:
            patterns.append({"label": "PROCEDURE", "pattern": term})

        # Dosage: number + unit (e.g. "875 mg", "40mg")
        for unit in DOSAGES_UNITS:
            patterns.append({
                "label": "DOSAGE",
                "pattern": [
                    {"TEXT": {"REGEX": r"^\d+(\.\d+)?$"}},
                    {"LOWER": unit},
                ],
            })

        ruler.add_patterns(patterns)

    def _extract_soap_text(self, text: str) -> str:
        """
        Pull clean readable text from the LLM output.
        Handles both raw JSON output and plain text.
        """
        # Try parsing as JSON first
        match = re.search(r'\{[\s\S]+\}', text)
        if match:
            try:
                data = json.loads(match.group())
                soap = data.get("soap_note", {})
                parts = [
                    soap.get("subjective", ""),
                    soap.get("objective", ""),
                    soap.get("assessment", ""),
                    soap.get("plan", ""),
                    data.get("image_findings_summary", ""),
                ]
                return " ".join(p for p in parts if p)
            except (json.JSONDecodeError, AttributeError):
                pass

        # Fallback: strip JSON keys and return readable text
        clean = re.sub(r'"[a-z_]+":\s*', '', text)
        clean = re.sub(r'[{}\[\]]', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()

    def extract(self, text: str) -> dict:
        """
        Extract clinical entities from LLM output text.
        Automatically cleans JSON wrapper before processing.
        """
        if self._nlp is None:
            self.load()

        clean_text = self._extract_soap_text(text)
        doc = self._nlp(clean_text)

        buckets: dict[str, list[str]] = {
            "symptoms": [], "diagnoses": [], "medications": [],
            "dosages": [], "procedures": [], "anatomy": [],
        }
        spans = []

        for ent in doc.ents:
            category = ENTITY_LABEL_MAP.get(ent.label_)
            if category:
                buckets[category].append(ent.text)
                spans.append({
                    "text": ent.text,
                    "label": category,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })

        result = {k: deduplicate_entities(v) for k, v in buckets.items()}
        result["drug_interactions"] = check_drug_interactions(
            result["medications"], self._drug_db
        )
        result["entity_spans"] = spans
        result["_clean_text"] = clean_text   # used by highlight_html
        return result

    def highlight_html(self, text: str, entities: dict) -> str:
        """Return clean SOAP text with entity spans highlighted."""
        # Use the already-cleaned text stored during extract()
        display_text = entities.get("_clean_text", self._extract_soap_text(text))
        spans = sorted(entities.get("entity_spans", []), key=lambda x: x["start"])

        COLOR_MAP = {
            "symptoms":   "#FFEB3B",
            "diagnoses":  "#F44336",
            "medications": "#4CAF50",
            "dosages":    "#2196F3",
            "anatomy":    "#9C27B0",
            "procedures": "#FF9800",
        }

        result = []
        prev = 0
        for span in spans:
            s, e = span["start"], span["end"]
            if s < prev:
                continue
            result.append(display_text[prev:s])
            color = COLOR_MAP.get(span["label"], "#ccc")
            result.append(
                f'<mark style="background:{color};color:#000;border-radius:3px;'
                f'padding:1px 4px;margin:0 1px" title="{span["label"]}">'
                f'{display_text[s:e]}</mark>'
            )
            prev = e
        result.append(display_text[prev:])
        return "".join(result)


_extractor: Optional[ClinicalNERExtractor] = None


def get_entity_extractor() -> ClinicalNERExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ClinicalNERExtractor()
        _extractor.load()
    return _extractor
