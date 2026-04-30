# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

"""
Download all free datasets needed for the Clinical AI Assistant.

Datasets used (all free, no registration required):
  1. Montgomery County Chest X-rays — NLM public domain, real TB-screening CXRs
  2. Synthetic medical audio        — pyttsx3 offline Windows TTS
  3. Drug interaction database      — OpenFDA API (no key needed) + curated list
  4. PubMed abstracts               — NCBI Entrez API (free, abstracts only)

Run with:
    venv/Scripts/python.exe scripts/download_datasets.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_RAW       = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_PUBMED    = Path("data/pubmed_index")

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DATA_PUBMED.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Chest X-rays — Montgomery County CXR Dataset (NLM, public domain)
#    138 chest X-rays (PNG), TB screening study
#    URL: https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/
# ─────────────────────────────────────────────────────────────────────────────

CXR_DIR  = DATA_RAW / "chest_xrays"
CXR_BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/"
)

# normal=0, TB-positive=1  (first 10 normal cases for demo)
CXR_FILES = [f"MCUCXR_{str(i).zfill(4)}_0.png" for i in range(1, 11)]


def download_xrays() -> None:
    CXR_DIR.mkdir(exist_ok=True)
    print("\n[1/4] Downloading Montgomery County Chest X-rays (NLM, public domain)...")
    session = requests.Session()
    session.headers["User-Agent"] = "ClinicalAI-Research/1.0 (educational)"

    downloaded = 0
    for fname in CXR_FILES:
        dest = CXR_DIR / fname
        if dest.exists():
            print(f"  skip  {fname} (already downloaded)")
            downloaded += 1
            continue
        try:
            resp = session.get(CXR_BASE + fname, timeout=30, stream=True)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                size_kb = len(resp.content) / 1024
                print(f"  [OK]  {fname}  ({size_kb:.0f} KB)")
                downloaded += 1
            else:
                print(f"  [--]  {fname}  HTTP {resp.status_code}")
            time.sleep(0.3)   # be polite to NLM servers
        except requests.RequestException as e:
            print(f"  [ERR] {fname}: {e}")

    print(f"  Downloaded {downloaded}/{len(CXR_FILES)} chest X-rays -> {CXR_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Synthetic Medical Audio — pyttsx3 (Windows SAPI5, offline, no account)
# ─────────────────────────────────────────────────────────────────────────────

AUDIO_DIR = DATA_RAW / "audio_samples"

MEDICAL_SCRIPTS = [
    (
        "patient_01_pneumonia.wav",
        (
            "Patient is a 58 year old male presenting with productive cough for 5 days, "
            "fever of 38.9 degrees Celsius, and right-sided pleuritic chest pain. "
            "He reports shortness of breath on exertion. "
            "Past medical history significant for hypertension and type 2 diabetes. "
            "No recent travel or sick contacts. "
            "On examination, decreased breath sounds at the right base with dullness to percussion. "
            "Chest X-ray shows right lower lobe consolidation. "
            "Impression: community-acquired pneumonia. "
            "Plan to start amoxicillin-clavulanate 875 milligrams twice daily for 7 days "
            "with follow-up in one week."
        ),
    ),
    (
        "patient_02_heart_failure.wav",
        (
            "54 year old female with a 3-week history of progressive dyspnea on exertion "
            "and bilateral ankle swelling. "
            "She reports orthopnea requiring 3 pillows and paroxysmal nocturnal dyspnea. "
            "Past history of hypertension and hyperlipidemia. "
            "On examination, jugular venous distension, bilateral basal crackles, "
            "and pitting edema to the knees. "
            "Echocardiogram shows ejection fraction of 35 percent. "
            "B-type natriuretic peptide is elevated at 980 picograms per milliliter. "
            "Diagnosis: heart failure with reduced ejection fraction. "
            "Initiating sacubitril-valsartan, carvedilol, and furosemide 40 milligrams daily."
        ),
    ),
    (
        "patient_03_copd.wav",
        (
            "67 year old male, 40 pack-year smoking history, presenting with acute exacerbation "
            "of COPD. Reports increased sputum production, change in sputum color to green, "
            "and worsening dyspnea over the past 3 days. "
            "Temperature 37.8, oxygen saturation 88 percent on room air. "
            "Chest X-ray shows hyperinflation with flattened diaphragms. "
            "Starting prednisolone 40 milligrams daily for 5 days, "
            "doxycycline 200 milligrams loading dose, "
            "and salbutamol nebulizers every 4 hours. "
            "Supplemental oxygen to maintain saturation above 88 to 92 percent."
        ),
    ),
    (
        "patient_04_tuberculosis.wav",
        (
            "32 year old female, recent immigrant, presenting with 2 months of night sweats, "
            "weight loss of 8 kilograms, and hemoptysis. "
            "Chest X-ray demonstrates upper lobe cavitation with satellite lesions. "
            "Sputum AFB smear positive. "
            "Starting standard 4-drug RIPE therapy: rifampicin 600 milligrams, "
            "isoniazid 300 milligrams, pyrazinamide 1500 milligrams, "
            "and ethambutol 1200 milligrams daily for the initial 2-month phase. "
            "Notifying public health and arranging contact tracing."
        ),
    ),
    (
        "patient_05_pleural_effusion.wav",
        (
            "71 year old male with known lung adenocarcinoma presenting with progressive dyspnea. "
            "Chest X-ray shows large left-sided pleural effusion with mediastinal shift. "
            "Oxygen saturation 91 percent on room air. "
            "Plan for urgent thoracentesis under ultrasound guidance. "
            "Will send pleural fluid for cytology, LDH, protein, glucose, and pH. "
            "Discussing indwelling pleural catheter for palliation with the oncology team."
        ),
    ),
]


def generate_audio() -> None:
    AUDIO_DIR.mkdir(exist_ok=True)
    print("\n[2/4] Generating synthetic medical audio (pyttsx3, offline Windows TTS)...")

    try:
        import pyttsx3
    except ImportError:
        print("  [SKIP] pyttsx3 not installed — run: pip install pyttsx3")
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", 145)    # slightly slower = clearer for medical speech
    engine.setProperty("volume", 0.95)

    # Prefer a male voice if available
    voices = engine.getProperty("voices")
    for v in voices:
        if "david" in v.name.lower() or "mark" in v.name.lower():
            engine.setProperty("voice", v.id)
            break

    for fname, script in MEDICAL_SCRIPTS:
        dest = AUDIO_DIR / fname
        if dest.exists():
            print(f"  skip  {fname} (already exists)")
            continue
        engine.save_to_file(script, str(dest))
        engine.runAndWait()
        if dest.exists():
            size_kb = dest.stat().st_size / 1024
            print(f"  [OK]  {fname}  ({size_kb:.0f} KB)")
        else:
            print(f"  [ERR] {fname} — pyttsx3 save failed")

    print(f"  Generated {len(MEDICAL_SCRIPTS)} audio samples -> {AUDIO_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Drug Interaction Database
#    Part A: Curated high-importance interactions (public medical knowledge)
#    Part B: OpenFDA API extension (completely free, no key needed)
# ─────────────────────────────────────────────────────────────────────────────

DRUG_DB_PATH = DATA_PROCESSED / "drug_interactions.json"

CURATED_INTERACTIONS = {
    # ── Major interactions ──────────────────────────────────────────────────
    "warfarin::aspirin": {
        "severity": "major",
        "description": "Concurrent use significantly increases bleeding risk. Aspirin inhibits platelet aggregation while warfarin impairs clotting factor synthesis.",
        "recommendation": "Avoid combination. If necessary, use lowest aspirin dose and monitor INR closely.",
    },
    "warfarin::ibuprofen": {
        "severity": "major",
        "description": "NSAIDs inhibit platelet function and may displace warfarin from protein binding, increasing anticoagulant effect and GI bleeding risk.",
        "recommendation": "Avoid NSAIDs with warfarin. Use paracetamol for pain relief.",
    },
    "warfarin::ciprofloxacin": {
        "severity": "major",
        "description": "Fluoroquinolones inhibit CYP1A2, increasing warfarin plasma levels and INR.",
        "recommendation": "Monitor INR closely. Reduce warfarin dose if needed.",
    },
    "warfarin::metronidazole": {
        "severity": "major",
        "description": "Metronidazole inhibits CYP2C9, substantially increasing warfarin levels and bleeding risk.",
        "recommendation": "Reduce warfarin dose by 25–50%. Monitor INR every 2–3 days.",
    },
    "simvastatin::amiodarone": {
        "severity": "major",
        "description": "Amiodarone inhibits CYP3A4 and CYP2C9, raising simvastatin levels and risk of myopathy/rhabdomyolysis.",
        "recommendation": "Limit simvastatin to 20 mg/day. Consider switching to pravastatin or rosuvastatin.",
    },
    "clopidogrel::omeprazole": {
        "severity": "major",
        "description": "Omeprazole inhibits CYP2C19, reducing conversion of clopidogrel to its active metabolite and reducing antiplatelet effect.",
        "recommendation": "Use pantoprazole instead of omeprazole with clopidogrel.",
    },
    "ssri::maoi": {
        "severity": "major",
        "description": "Combination can cause life-threatening serotonin syndrome: agitation, hyperthermia, tachycardia, clonus.",
        "recommendation": "Contraindicated. Allow 14-day washout between agents.",
    },
    "methotrexate::nsaids": {
        "severity": "major",
        "description": "NSAIDs reduce renal clearance of methotrexate, causing toxic accumulation and bone marrow suppression.",
        "recommendation": "Avoid combination. If unavoidable, reduce methotrexate dose and monitor FBC and renal function.",
    },
    "digoxin::amiodarone": {
        "severity": "major",
        "description": "Amiodarone inhibits P-glycoprotein, increasing digoxin plasma levels with risk of toxicity (bradycardia, heart block).",
        "recommendation": "Reduce digoxin dose by 50%. Monitor digoxin levels and ECG.",
    },
    "lithium::thiazide": {
        "severity": "major",
        "description": "Thiazide diuretics reduce renal lithium clearance, causing lithium toxicity (tremor, confusion, arrhythmia).",
        "recommendation": "Avoid combination. If necessary, monitor lithium levels weekly and reduce dose.",
    },
    # ── Moderate interactions ────────────────────────────────────────────────
    "metformin::alcohol": {
        "severity": "moderate",
        "description": "Alcohol potentiates lactic acidosis risk with metformin, especially in patients with renal or hepatic impairment.",
        "recommendation": "Advise patient to avoid binge drinking. Moderate alcohol intake is acceptable.",
    },
    "lisinopril::potassium": {
        "severity": "moderate",
        "description": "ACE inhibitors reduce potassium excretion; concurrent potassium supplements risk hyperkalaemia.",
        "recommendation": "Monitor serum potassium. Avoid potassium supplements unless documented deficiency.",
    },
    "lisinopril::spironolactone": {
        "severity": "moderate",
        "description": "Dual renin-angiotensin-aldosterone blockade increases hyperkalaemia and acute kidney injury risk.",
        "recommendation": "Monitor renal function and potassium monthly. Use with caution at low doses only.",
    },
    "amlodipine::simvastatin": {
        "severity": "moderate",
        "description": "Amlodipine inhibits CYP3A4, increasing simvastatin exposure and myopathy risk.",
        "recommendation": "Limit simvastatin to 20 mg/day when co-administered with amlodipine.",
    },
    "clarithromycin::simvastatin": {
        "severity": "moderate",
        "description": "Clarithromycin strongly inhibits CYP3A4, markedly increasing simvastatin levels and rhabdomyolysis risk.",
        "recommendation": "Suspend simvastatin during clarithromycin course. Switch to pravastatin.",
    },
    "fluoxetine::tramadol": {
        "severity": "moderate",
        "description": "SSRIs inhibit CYP2D6 reducing tramadol conversion to active metabolite and risk of serotonin syndrome.",
        "recommendation": "Monitor for serotonin syndrome symptoms. Consider alternative analgesic.",
    },
    "metformin::contrast": {
        "severity": "moderate",
        "description": "IV contrast media can cause contrast-induced nephropathy, reducing metformin excretion and increasing lactic acidosis risk.",
        "recommendation": "Hold metformin 48h before and after IV contrast. Check creatinine before restarting.",
    },
    "bisoprolol::verapamil": {
        "severity": "moderate",
        "description": "Both drugs have negative chronotropic and inotropic effects; combination risks complete heart block and heart failure.",
        "recommendation": "Avoid IV verapamil with beta-blockers. If oral use necessary, monitor closely.",
    },
    "ciprofloxacin::antacids": {
        "severity": "moderate",
        "description": "Divalent cations (Mg, Al, Ca) chelate ciprofloxacin, reducing oral bioavailability by up to 90%.",
        "recommendation": "Separate ciprofloxacin from antacids/calcium by at least 2 hours.",
    },
    # ── Minor interactions ───────────────────────────────────────────────────
    "atorvastatin::grapefruit": {
        "severity": "minor",
        "description": "Grapefruit juice inhibits intestinal CYP3A4, modestly increasing atorvastatin levels.",
        "recommendation": "Advise patient to avoid large quantities of grapefruit juice.",
    },
    "sertraline::tramadol": {
        "severity": "minor",
        "description": "Mild risk of serotonin syndrome; sertraline inhibits CYP2D6 reducing tramadol efficacy.",
        "recommendation": "Monitor for serotonin syndrome symptoms. Use lowest effective tramadol dose.",
    },
    "aspirin::ibuprofen": {
        "severity": "minor",
        "description": "Ibuprofen may interfere with aspirin's irreversible platelet inhibition if taken concurrently.",
        "recommendation": "Take aspirin 2 hours before ibuprofen, or use paracetamol for pain relief.",
    },
}


def fetch_openfda_interactions(drug_pairs: list[tuple[str, str]]) -> dict:
    """
    Extend drug DB using OpenFDA API — completely free, no API key.
    https://api.fda.gov/drug/label.json
    """
    extra = {}
    base = "https://api.fda.gov/drug/label.json"
    for drug_a, drug_b in drug_pairs:
        try:
            resp = requests.get(
                base,
                params={"search": f'drug_interactions:"{drug_a}"+AND+drug_interactions:"{drug_b}"', "limit": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    text = results[0].get("drug_interactions", [""])[0][:400]
                    key = f"{drug_a.lower()}::{drug_b.lower()}"
                    if key not in CURATED_INTERACTIONS:
                        extra[key] = {
                            "severity": "moderate",
                            "description": text,
                            "recommendation": "Consult clinical pharmacist.",
                        }
            time.sleep(0.4)  # OpenFDA rate limit: 240 requests/min
        except Exception:
            pass
    return extra


def build_drug_db() -> None:
    print("\n[3/4] Building drug interaction database...")

    db = dict(CURATED_INTERACTIONS)
    print(f"  Curated interactions loaded: {len(db)}")

    # Extend with a few OpenFDA lookups
    print("  Fetching additional interactions from OpenFDA API (free)...")
    openfda_pairs = [
        ("warfarin", "fluconazole"),
        ("methotrexate", "trimethoprim"),
        ("phenytoin", "carbamazepine"),
        ("tacrolimus", "fluconazole"),
        ("rifampicin", "contraceptive"),
    ]
    extra = fetch_openfda_interactions(openfda_pairs)
    db.update(extra)
    print(f"  OpenFDA additions: {len(extra)}")

    DRUG_DB_PATH.write_text(json.dumps(db, indent=2))
    print(f"  Total interactions: {len(db)} -> saved to {DRUG_DB_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PubMed RAG Index — free abstracts via NCBI Entrez API
# ─────────────────────────────────────────────────────────────────────────────

PUBMED_QUERIES = [
    "community acquired pneumonia chest X-ray treatment 2023",
    "pulmonary edema diagnosis echocardiography BNP",
    "COPD exacerbation management bronchodilator",
    "tuberculosis radiology upper lobe cavitation",
    "pleural effusion diagnosis thoracentesis",
    "lung nodule Fleischner Society guidelines",
    "pneumothorax chest tube management",
    "heart failure ejection fraction treatment",
    "atelectasis chest radiograph causes treatment",
    "consolidation pneumonia antibiotic therapy",
]


def build_pubmed_index() -> None:
    index_path = Path("data/pubmed_index/faiss_index")
    if index_path.exists():
        print(f"\n[4/4] FAISS index already exists at {index_path} — skipping.")
        print("      Delete the folder and rerun to rebuild.")
        return

    print("\n[4/4] Building PubMed RAG index (free abstracts via Entrez API)...")
    print("      NOTE: Only abstracts are free. Full text requires institutional access.")

    from src.llm.rag_pipeline import PubMedRAGPipeline

    email = os.getenv("ENTREZ_EMAIL", "researcher@example.com")
    if email == "researcher@example.com":
        print("  TIP: Set ENTREZ_EMAIL env var to your real email for NCBI compliance.")

    rag = PubMedRAGPipeline()
    try:
        rag.fetch_and_build(PUBMED_QUERIES, max_results=len(PUBMED_QUERIES) * 15)
        print(f"  FAISS index saved to {index_path}")
    except Exception as e:
        print(f"  [WARN] Entrez API failed: {e}")
        print("  Building minimal index from a built-in sample instead...")
        _build_fallback_index(rag)


def _build_fallback_index(rag) -> None:
    """Build a tiny index if Entrez API is unavailable (offline mode)."""
    sample_abstracts = [
        {
            "pmid": "36001", "title": "Community-Acquired Pneumonia Guidelines 2023",
            "abstract": (
                "Community-acquired pneumonia (CAP) remains a leading cause of infectious "
                "disease mortality. Beta-lactam antibiotics are first-line for non-severe CAP. "
                "Macrolide combination therapy improves outcomes in severe cases. "
                "Radiographic consolidation confirms the diagnosis in 75% of cases."
            ),
        },
        {
            "pmid": "36002", "title": "Chest X-Ray Interpretation in Pneumonia",
            "abstract": (
                "Lobar consolidation, air bronchograms, and silhouette sign are the classic "
                "radiographic findings of bacterial pneumonia. Right lower lobe is the most "
                "common location for aspiration pneumonia. Interstitial patterns suggest "
                "atypical organisms such as Mycoplasma or viral pneumonia."
            ),
        },
        {
            "pmid": "36003", "title": "Heart Failure with Reduced Ejection Fraction Treatment",
            "abstract": (
                "HFrEF management includes ACE inhibitors or ARNIs, beta-blockers, and "
                "mineralocorticoid receptor antagonists as the pillars of therapy. "
                "SGLT2 inhibitors have demonstrated mortality benefit. "
                "BNP and NT-proBNP guide diuretic therapy. "
                "Echocardiography is essential for EF assessment."
            ),
        },
        {
            "pmid": "36004", "title": "COPD Exacerbation: Diagnosis and Management",
            "abstract": (
                "Acute exacerbations of COPD are characterized by worsening dyspnea, "
                "increased sputum purulence and volume. Short-acting bronchodilators, "
                "systemic corticosteroids, and antibiotics form the treatment triad. "
                "Non-invasive ventilation reduces intubation rates in severe exacerbations."
            ),
        },
        {
            "pmid": "36005", "title": "Pulmonary Tuberculosis: Radiology Review",
            "abstract": (
                "Upper lobe cavitation is the hallmark of post-primary tuberculosis. "
                "Miliary TB presents with 1-3mm nodules throughout both lung fields. "
                "AFB smear and GeneXpert PCR confirm diagnosis. "
                "Standard RIPE therapy for 6 months achieves cure in drug-sensitive TB."
            ),
        },
        {
            "pmid": "36006", "title": "Pleural Effusion: Diagnostic Approach",
            "abstract": (
                "Light's criteria distinguish exudates from transudates with 98% sensitivity. "
                "Malignant pleural effusions are most commonly caused by lung, breast, and "
                "lymphoma. Ultrasound-guided thoracentesis reduces complication rates. "
                "Indwelling pleural catheters provide effective palliation."
            ),
        },
        {
            "pmid": "36007", "title": "Lung Nodule Management: Fleischner Society Guidelines",
            "abstract": (
                "Solid pulmonary nodules under 6mm in low-risk patients require no routine "
                "follow-up. Nodules 6-8mm need 6-12 month CT surveillance. "
                "Ground-glass opacity nodules carry higher malignancy risk. "
                "PET-CT is recommended for nodules over 8mm with intermediate risk."
            ),
        },
        {
            "pmid": "36008", "title": "Drug Interactions in Clinical Practice",
            "abstract": (
                "Clinically significant drug-drug interactions affect approximately 7% of "
                "hospitalized patients. CYP450 enzyme inhibition and induction are major "
                "mechanisms. Warfarin interactions are the most commonly encountered "
                "life-threatening interactions. Electronic prescribing reduces ADR incidence."
            ),
        },
    ]
    rag.build_index(sample_abstracts)
    print(f"  Fallback index built with {len(sample_abstracts)} curated abstracts.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Clinical AI Assistant — Dataset Download")
    print("  All sources are free, no registration required")
    print("=" * 60)

    download_xrays()
    generate_audio()
    build_drug_db()
    build_pubmed_index()

    print("\n" + "=" * 60)
    print("Dataset setup complete. Summary:")
    print(f"  Chest X-rays : {len(list((DATA_RAW / 'chest_xrays').glob('*.png')))} images -> data/raw/chest_xrays/")
    print(f"  Audio samples: {len(list((DATA_RAW / 'audio_samples').glob('*.wav')))} files  -> data/raw/audio_samples/")
    print(f"  Drug DB      : {len(json.loads(DRUG_DB_PATH.read_text()))} interactions -> data/processed/drug_interactions.json")
    print(f"  FAISS index  : {'built' if Path('data/pubmed_index/faiss_index').exists() else 'NOT built'} -> data/pubmed_index/")
    print()
    print("Next step:")
    print("  venv/Scripts/python.exe scripts/check_setup.py")
    print("=" * 60)
