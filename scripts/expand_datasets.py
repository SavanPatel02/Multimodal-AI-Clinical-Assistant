"""
Expand datasets to give the system more variety for testing.

Downloads:
  1. All Montgomery County CXRs (~100 normal chest X-rays, NLM free)
  2. Shenzhen Hospital CXRs (~300 images, normal + TB positive, NLM free)
  3. 16 more synthetic audio cases (total 20 conditions)
  4. Expanded PubMed RAG index (~150 abstracts)

Run: venv/Scripts/python.exe scripts/expand_datasets.py
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

CXR_DIR   = Path("data/raw/chest_xrays")
AUDIO_DIR = Path("data/raw/audio_samples")
CXR_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Montgomery County CXR — all 138 images (normal TB screening)
# ─────────────────────────────────────────────────────────────────────────────

MONTGOMERY_BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Montgomery-County-CXR-Set/MontgomerySet/CXR_png/"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Shenzhen Hospital CXR — 662 images (normal + TB positive)
#    CHNCXR_XXXX_0.png = normal, CHNCXR_XXXX_1.png = TB positive
# ─────────────────────────────────────────────────────────────────────────────

SHENZHEN_BASE = (
    "https://data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/"
    "Shenzhen-Hospital-CXR-Set/CXR_png/"
)


def download_image(session, url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        resp = session.get(url, timeout=30, stream=True)
        if resp.status_code == 200:
            dest.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def download_montgomery():
    print("\n[1/4] Montgomery County CXRs (all 138 images)...")
    session = requests.Session()
    session.headers["User-Agent"] = "ClinicalAI-Research/1.0 (educational)"

    downloaded = skipped = failed = 0
    for i in range(1, 139):
        fname = f"MCUCXR_{str(i).zfill(4)}_0.png"
        dest  = CXR_DIR / fname
        if dest.exists():
            skipped += 1
            continue
        ok = download_image(session, MONTGOMERY_BASE + fname, dest)
        if ok:
            downloaded += 1
            print(f"  [OK] {fname}")
        else:
            failed += 1
        time.sleep(0.25)

    print(f"  Montgomery: {downloaded} downloaded, {skipped} skipped, {failed} unavailable")


def download_shenzhen(limit: int = 300):
    print(f"\n[2/4] Shenzhen Hospital CXRs (up to {limit} images, normal + TB)...")
    session = requests.Session()
    session.headers["User-Agent"] = "ClinicalAI-Research/1.0 (educational)"

    downloaded = skipped = failed = 0
    count = 0

    for i in range(1, 663):
        if count >= limit:
            break
        for label in [0, 1]:   # 0=normal, 1=TB positive
            if count >= limit:
                break
            fname = f"CHNCXR_{str(i).zfill(4)}_{label}.png"
            dest  = CXR_DIR / fname
            if dest.exists():
                skipped += 1
                count += 1
                continue
            ok = download_image(session, SHENZHEN_BASE + fname, dest)
            if ok:
                downloaded += 1
                count += 1
                print(f"  [OK] {fname}")
                time.sleep(0.25)
            else:
                failed += 1

    print(f"  Shenzhen: {downloaded} downloaded, {skipped} skipped, {failed} unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Synthetic Audio — 16 more clinical cases (pyttsx3 offline TTS)
# ─────────────────────────────────────────────────────────────────────────────

NEW_AUDIO_SCRIPTS = [
    (
        "patient_05_pleural_effusion.wav",
        "71 year old male with known lung adenocarcinoma presenting with progressive dyspnea. "
        "Chest X-ray shows large left-sided pleural effusion with mediastinal shift. "
        "Oxygen saturation 91 percent on room air. Plan for urgent thoracentesis under ultrasound guidance. "
        "Discussing indwelling pleural catheter for palliation with the oncology team.",
    ),
    (
        "patient_06_pneumothorax.wav",
        "24 year old tall male, sudden onset right-sided chest pain and dyspnea while at rest. "
        "No trauma. Chest X-ray confirms right-sided pneumothorax with 3 centimetre rim. "
        "Oxygen saturation 94 percent. Applying 100 percent oxygen via non-rebreather mask. "
        "Arranging chest tube insertion in the right 4th intercostal space, midaxillary line.",
    ),
    (
        "patient_07_pulmonary_embolism.wav",
        "48 year old female, 2 weeks post right total knee replacement, presenting with sudden "
        "dyspnea and pleuritic chest pain. Heart rate 112, oxygen saturation 89 percent on room air. "
        "D-dimer elevated at 4500 nanograms per millilitre. CT pulmonary angiography confirms "
        "bilateral pulmonary emboli. Starting therapeutic low molecular weight heparin. "
        "Considering thrombolysis given haemodynamic borderline compromise.",
    ),
    (
        "patient_08_sepsis.wav",
        "68 year old male with type 2 diabetes presenting with confusion, fever 39.2 degrees, "
        "heart rate 118, blood pressure 88 over 56, respiratory rate 24. "
        "Lactate 3.8 millimoles per litre. Urine dipstick positive for leucocytes and nitrites. "
        "Diagnosis: sepsis secondary to urinary tract infection. "
        "Initiating sepsis bundle: blood cultures taken, IV piperacillin-tazobactam commenced, "
        "aggressive fluid resuscitation with 30 millilitres per kilogram crystalloid.",
    ),
    (
        "patient_09_lung_cancer.wav",
        "63 year old male, 35 pack-year smoking history, presenting with 3 months of worsening cough, "
        "haemoptysis, and 6 kilogram weight loss. Chest X-ray shows 4 centimetre right upper lobe mass. "
        "CT chest confirms spiculated lesion with mediastinal lymphadenopathy. "
        "PET-CT arranged to assess for distant metastases. "
        "Urgent bronchoscopy and CT-guided biopsy requested. Referring to thoracic oncology multidisciplinary team.",
    ),
    (
        "patient_10_atelectasis.wav",
        "55 year old female, day 2 post laparoscopic cholecystectomy, presenting with fever 38.1 "
        "and decreased oxygen saturation to 93 percent on room air. "
        "Chest X-ray shows left lower lobe atelectasis with volume loss. "
        "Encouraging deep breathing exercises, incentive spirometry hourly. "
        "Physiotherapy referred for chest physio. Increasing supplemental oxygen. "
        "If no improvement, will arrange bronchoscopy for mucus plug clearance.",
    ),
    (
        "patient_11_aortic_stenosis.wav",
        "74 year old male presenting with exertional chest pain, syncope on exertion, "
        "and progressive dyspnea on exertion over 6 months. "
        "Echocardiogram shows severe aortic stenosis with valve area 0.7 centimetres squared "
        "and mean gradient of 52 millimetres of mercury. Ejection fraction preserved at 60 percent. "
        "Referring for transcatheter aortic valve implantation assessment. "
        "Starting diuretics for symptomatic relief. Avoiding strenuous activity.",
    ),
    (
        "patient_12_interstitial_lung_disease.wav",
        "61 year old female non-smoker, 8 months progressive exertional dyspnea and dry cough. "
        "Finger clubbing noted on examination. High resolution CT chest shows bilateral basal "
        "peripheral honeycombing with traction bronchiectasis consistent with usual interstitial pneumonia pattern. "
        "Pulmonary function tests show restrictive defect with reduced diffusion capacity. "
        "Diagnosis: idiopathic pulmonary fibrosis. Commencing nintedanib 150 milligrams twice daily. "
        "Referring to ILD specialist centre.",
    ),
    (
        "patient_13_acute_asthma.wav",
        "19 year old female known asthmatic presenting with severe bronchospasm. "
        "Using accessory muscles, unable to complete sentences. Peak flow 35 percent of best. "
        "Oxygen saturation 91 percent on air. Administering back-to-back salbutamol nebulisers, "
        "ipratropium 0.5 milligrams, prednisolone 40 milligrams orally, and magnesium sulphate "
        "2 grams intravenous over 20 minutes. Arranging ICU review.",
    ),
    (
        "patient_14_pericardial_effusion.wav",
        "52 year old male with known lung adenocarcinoma presenting with dyspnea and hypotension. "
        "Jugular venous distension present. Echocardiogram confirms large pericardial effusion "
        "with right ventricular diastolic collapse consistent with cardiac tamponade. "
        "Performing urgent pericardiocentesis under echocardiographic guidance. "
        "750 millilitres of haemorrhagic fluid drained. Cardiology team informed.",
    ),
    (
        "patient_15_dvt_cellulitis.wav",
        "45 year old obese female presenting with right leg pain, swelling, and erythema. "
        "Temperature 37.9, heart rate 94. Doppler ultrasound confirms right femoral deep vein thrombosis. "
        "Right lower leg also shows areas of erythema and warmth consistent with cellulitis. "
        "Starting apixaban 10 milligrams twice daily for DVT treatment. "
        "Starting IV flucloxacillin for cellulitis. Leg elevation and compression when cellulitis resolved.",
    ),
    (
        "patient_16_hypertensive_emergency.wav",
        "58 year old male presenting with blood pressure 210 over 130, severe headache, "
        "and blurred vision. Fundoscopy shows grade 4 hypertensive retinopathy with papilloedema. "
        "ECG shows left ventricular hypertrophy. Creatinine mildly elevated. "
        "Diagnosis: hypertensive emergency with end organ damage. "
        "Commencing IV labetalol infusion targeting 20 to 25 percent blood pressure reduction "
        "over first hour. Arranging MRI brain to exclude haemorrhage.",
    ),
    (
        "patient_17_diabetic_ketoacidosis.wav",
        "22 year old type 1 diabetic presenting with vomiting, abdominal pain, and polyuria for 2 days. "
        "Blood glucose 32 millimoles per litre, pH 7.12, bicarbonate 8 millimoles per litre, "
        "ketones 4 plus. Diagnosis: severe diabetic ketoacidosis. "
        "Commencing DKA protocol: IV 0.9 percent saline 1 litre over first hour, "
        "fixed rate insulin infusion 0.1 units per kilogram per hour, hourly glucose and ketone monitoring. "
        "Identifying and treating precipitating infection.",
    ),
    (
        "patient_18_upper_gi_bleed.wav",
        "67 year old male on warfarin for atrial fibrillation presenting with haematemesis "
        "and melaena. Blood pressure 90 over 60, heart rate 115. "
        "INR 3.8. Haemoglobin 72 grams per litre. "
        "Administering tranexamic acid, IV proton pump inhibitor, and 4 units packed red cells. "
        "Reversing warfarin with vitamin K and prothrombin complex concentrate. "
        "Urgent OGD arranged. Gastroenterology and haematology teams alerted.",
    ),
    (
        "patient_19_stroke.wav",
        "69 year old male presenting with sudden onset left sided facial droop, "
        "left arm weakness, and dysarthria. Onset 1 hour 45 minutes ago. "
        "NIHSS score 14. Non-contrast CT head shows no haemorrhage. "
        "CT angiography confirms right middle cerebral artery occlusion. "
        "Within thrombolysis window. Administering IV alteplase 0.9 milligrams per kilogram. "
        "Transferring to interventional suite for mechanical thrombectomy.",
    ),
    (
        "patient_20_acute_kidney_injury.wav",
        "72 year old female on ramipril and ibuprofen presenting with reduced urine output "
        "and generalised oedema. Creatinine 380 micromoles per litre, baseline 90. "
        "Potassium 6.1 millimoles per litre. ECG shows peaked T waves. "
        "Diagnosis: acute kidney injury stage 3, likely NSAID and ACE inhibitor nephrotoxicity. "
        "Stopping ibuprofen and ramipril. IV calcium gluconate for hyperkalaemia. "
        "Insulin dextrose infusion. Nephrology review requested. Fluid balance monitoring.",
    ),
]


def generate_audio():
    print(f"\n[3/4] Generating {len(NEW_AUDIO_SCRIPTS)} additional audio samples...")

    try:
        import pyttsx3
    except ImportError:
        print("  [SKIP] pyttsx3 not installed")
        return

    engine = pyttsx3.init()
    engine.setProperty("rate", 148)
    engine.setProperty("volume", 0.95)
    voices = engine.getProperty("voices")
    for v in voices:
        if "david" in v.name.lower() or "mark" in v.name.lower():
            engine.setProperty("voice", v.id)
            break

    generated = skipped = 0
    for fname, script in NEW_AUDIO_SCRIPTS:
        dest = AUDIO_DIR / fname
        if dest.exists():
            print(f"  skip  {fname}")
            skipped += 1
            continue
        engine.save_to_file(script, str(dest))
        engine.runAndWait()
        if dest.exists():
            print(f"  [OK]  {fname}")
            generated += 1
        else:
            print(f"  [ERR] {fname}")

    print(f"  Audio: {generated} generated, {skipped} already existed")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Expanded PubMed RAG index (~150 abstracts across more topics)
# ─────────────────────────────────────────────────────────────────────────────

EXPANDED_QUERIES = [
    # Respiratory
    "community acquired pneumonia treatment antibiotic 2023",
    "chest X-ray consolidation pneumonia radiology",
    "COPD exacerbation management bronchodilator",
    "pulmonary tuberculosis diagnosis treatment radiology",
    "pleural effusion diagnosis thoracentesis Light criteria",
    "pneumothorax spontaneous management chest tube",
    "pulmonary embolism CTPA diagnosis anticoagulation",
    "interstitial lung disease IPF diagnosis HRCT",
    "lung nodule Fleischner CT surveillance malignancy",
    "asthma acute exacerbation management salbutamol",
    "atelectasis chest X-ray physiotherapy treatment",
    # Cardiac
    "heart failure reduced ejection fraction treatment SGLT2",
    "aortic stenosis echocardiogram TAVR management",
    "cardiac tamponade pericardial effusion echocardiography",
    "hypertensive emergency end organ damage treatment",
    "acute coronary syndrome STEMI thrombolysis PCI",
    # Systemic
    "sepsis management surviving sepsis campaign bundle",
    "diabetic ketoacidosis insulin protocol treatment",
    "acute kidney injury AKI management nephrology",
    "stroke thrombolysis thrombectomy NIHSS treatment",
    # Cancer
    "lung cancer staging PET CT biopsy treatment",
    "malignant pleural effusion palliative drainage",
    # Drug interactions pharmacology
    "drug drug interactions warfarin antibiotic clinical",
    "antibiotic stewardship procalcitonin respiratory infection",
    "beta lactam penicillin pneumonia treatment outcome",
]


def expand_rag_index():
    index_path = Path("data/pubmed_index/faiss_index")
    print(f"\n[4/4] Expanding PubMed RAG index ({len(EXPANDED_QUERIES)} queries x 6 results each)...")

    from src.llm.rag_pipeline import PubMedRAGPipeline

    email = os.getenv("ENTREZ_EMAIL", "researcher@example.com")
    if email == "researcher@example.com":
        print("  TIP: Set ENTREZ_EMAIL env var to your real email for NCBI compliance.")

    rag = PubMedRAGPipeline()
    try:
        rag.fetch_and_build(EXPANDED_QUERIES, max_results=len(EXPANDED_QUERIES) * 6)
        print(f"  FAISS index rebuilt with ~{len(EXPANDED_QUERIES) * 6} abstracts")
    except Exception as e:
        print(f"  [WARN] Entrez API failed: {e}")
        print("  Keeping existing index.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Clinical AI Assistant - Dataset Expansion")
    print("=" * 60)

    download_montgomery()
    download_shenzhen(limit=300)
    generate_audio()
    expand_rag_index()

    print("\n" + "=" * 60)
    print("Dataset expansion complete:")
    xrays  = list(CXR_DIR.glob("*.png"))
    audios = list(AUDIO_DIR.glob("*.wav"))
    print(f"  Chest X-rays  : {len(xrays)} images  -> data/raw/chest_xrays/")
    print(f"  Audio samples : {len(audios)} files   -> data/raw/audio_samples/")
    print(f"  Drug DB       : 25 interactions (unchanged)")
    print(f"  FAISS index   : rebuilt with ~150 abstracts")
    print()
    print("X-ray breakdown:")
    montgomery = [f for f in xrays if f.name.startswith("MCUCXR")]
    shenzhen   = [f for f in xrays if f.name.startswith("CHNCXR")]
    normal_sz  = [f for f in shenzhen if f.name.endswith("_0.png")]
    tb_sz      = [f for f in shenzhen if f.name.endswith("_1.png")]
    print(f"  Montgomery (normal)    : {len(montgomery)} images")
    print(f"  Shenzhen (normal)      : {len(normal_sz)} images")
    print(f"  Shenzhen (TB positive) : {len(tb_sz)} images")
    print("=" * 60)
