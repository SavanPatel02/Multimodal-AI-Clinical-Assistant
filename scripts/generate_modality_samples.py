"""
Generate sample images for MRI, CT Scan, and Ultrasound modalities,
and generate all remaining audio files.

Run: venv/Scripts/python.exe scripts/generate_modality_samples.py
"""
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

MRI_DIR   = Path("data/raw/mri_scans")
CT_DIR    = Path("data/raw/ct_scans")
US_DIR    = Path("data/raw/ultrasound")
AUDIO_DIR = Path("data/raw/audio_samples")

for d in [MRI_DIR, CT_DIR, US_DIR, AUDIO_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1.  BRAIN MRI  (T1-weighted appearance)
# =============================================================================

def make_brain_mri(
    case_name: str,
    size: int = 512,
    has_lesion: bool = False,
    lesion_pos: tuple = (0.4, 0.35),
    lesion_r: int = 30,
    seed: int = 0,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2

    # Skull (bright white ring)
    skull_r, skull_thick = int(size * 0.44), int(size * 0.04)
    for y in range(size):
        for x in range(size):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if skull_r - skull_thick < d < skull_r:
                img[y, x] = 0.85 + rng.random() * 0.15

    # White matter (mid-grey)
    wm_r = skull_r - skull_thick - 2
    Y, X = np.ogrid[:size, :size]
    mask_brain = (X - cx) ** 2 + (Y - cy) ** 2 < wm_r ** 2
    img[mask_brain] = 0.45 + rng.random((size, size))[mask_brain] * 0.12

    # Grey matter cortex (slightly brighter band near skull)
    for y in range(size):
        for x in range(size):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if wm_r - int(size * 0.05) < d < wm_r:
                img[y, x] = 0.58 + rng.random() * 0.10

    # Ventricles (dark CSF spaces)
    vent_coords = [
        (cx - size // 10, cy - size // 14, size // 16, size // 10),
        (cx + size // 10, cy - size // 14, size // 16, size // 10),
    ]
    for vx, vy, vw, vh in vent_coords:
        rr, cc = np.ogrid[:size, :size]
        v_mask = ((rr - vy) ** 2 / vh ** 2 + (cc - vx) ** 2 / vw ** 2) < 1
        img[v_mask] = 0.08 + rng.random((size, size))[v_mask] * 0.05

    # Falx cerebri (thin bright midline)
    img[cy - wm_r:cy + wm_r, cx - 2:cx + 2] = np.clip(
        img[cy - wm_r:cy + wm_r, cx - 2:cx + 2] + 0.25, 0, 1
    )

    # Pathological lesion (tumour / infarct)
    if has_lesion:
        lx = int(lesion_pos[0] * size)
        ly = int(lesion_pos[1] * size)
        # Bright ring (enhancing tumour)
        for y in range(size):
            for x in range(size):
                d = math.sqrt((x - lx) ** 2 + (y - ly) ** 2)
                if d < lesion_r:
                    ring = 1 - d / lesion_r
                    img[y, x] = 0.15 + ring * 0.1 + rng.random() * 0.05
                if lesion_r - 8 < d < lesion_r:
                    img[y, x] = 0.9 + rng.random() * 0.1  # bright ring

    # Smooth and convert
    img = np.clip(img, 0, 1)
    arr = (img * 255).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (3, 3), 0.8)
    return Image.fromarray(arr).convert("RGB")


MRI_CASES = [
    ("brain_mri_normal_01.jpg",   False, (0.4, 0.35), 0,  1),
    ("brain_mri_normal_02.jpg",   False, (0.6, 0.4),  0,  2),
    ("brain_mri_normal_03.jpg",   False, (0.5, 0.45), 0,  3),
    ("brain_mri_tumour_01.jpg",   True,  (0.35, 0.3), 32, 4),
    ("brain_mri_tumour_02.jpg",   True,  (0.62, 0.38),28, 5),
    ("brain_mri_infarct_01.jpg",  True,  (0.58, 0.42),22, 6),
    ("brain_mri_infarct_02.jpg",  True,  (0.42, 0.55),18, 7),
    ("brain_mri_normal_04.jpg",   False, (0.5, 0.5),  0,  8),
    ("brain_mri_normal_05.jpg",   False, (0.48, 0.52),0,  9),
    ("brain_mri_tumour_03.jpg",   True,  (0.30, 0.45),35, 10),
]


def generate_mri():
    print("\n[1/3] Generating Brain MRI images...")
    for fname, has_lesion, lpos, lr, seed in MRI_CASES:
        dest = MRI_DIR / fname
        if dest.exists():
            print(f"  skip  {fname}")
            continue
        img = make_brain_mri(fname, size=512, has_lesion=has_lesion,
                             lesion_pos=lpos, lesion_r=lr, seed=seed)
        img.save(str(dest), quality=95)
        tag = "tumour" if has_lesion and lr > 25 else "infarct" if has_lesion else "normal"
        print(f"  [OK]  {fname}  ({tag})")
    print(f"  MRI: {len(list(MRI_DIR.glob('*.jpg')))} images -> {MRI_DIR}")


# =============================================================================
# 2.  CT CHEST SCAN  (lung window appearance)
# =============================================================================

def make_ct_chest(
    case_name: str,
    size: int = 512,
    pathology: str = "normal",
    seed: int = 0,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, int(size * 0.52)

    # Body wall (soft tissue — mid-grey oval)
    body_rx, body_ry = int(size * 0.46), int(size * 0.40)
    Y, X = np.ogrid[:size, :size]
    body_mask = ((X - cx) ** 2 / body_rx ** 2 + (Y - cy) ** 2 / body_ry ** 2) < 1
    img[body_mask] = 0.35 + rng.random((size, size))[body_mask] * 0.08

    # Rib cage (bright dots arranged in arc)
    for side in [-1, 1]:
        for i in range(8):
            angle = math.radians(30 + i * 15)
            rx = int(cx + side * body_rx * 0.82 * math.cos(angle))
            ry = int(cy - body_ry * 0.75 * math.sin(angle))
            cv2.circle(img, (rx, ry), 8, 0.88, -1)

    # Lung fields (very dark — air)
    for side, lx_off in [(-1, -int(size * 0.18)), (1, int(size * 0.18))]:
        lx = cx + lx_off
        ly = cy - int(size * 0.02)
        lrx, lry = int(size * 0.16), int(size * 0.22)
        lung_mask = ((X - lx) ** 2 / lrx ** 2 + (Y - ly) ** 2 / lry ** 2) < 1
        img[lung_mask] = 0.05 + rng.random((size, size))[lung_mask] * 0.04

        # Pulmonary vessels (thin bright lines)
        for _ in range(12):
            sx = lx + rng.integers(-lrx + 10, lrx - 10)
            sy = ly + rng.integers(-lry + 10, lry - 10)
            ex = lx + rng.integers(-lrx + 10, lrx - 10)
            ey = ly + rng.integers(-lry + 10, lry - 10)
            cv2.line(img, (sx, sy), (ex, ey), 0.55, 1)

    # Mediastinum (heart / trachea — bright centre)
    med_mask = ((X - cx) ** 2 / (size // 10) ** 2 + (Y - (cy + size // 20)) ** 2 / (size // 7) ** 2) < 1
    img[med_mask] = 0.55 + rng.random((size, size))[med_mask] * 0.1

    # Spine (very bright posterior midline)
    spine_mask = ((X - cx) ** 2 < (size // 25) ** 2) & ((Y - (cy + int(size * 0.18))) ** 2 < (size // 20) ** 2)
    img[spine_mask] = 0.92

    # Pathology overlay
    if pathology == "consolidation":
        # Right lower lobe consolidation — white fill in right lung
        rng2 = np.random.default_rng(seed + 100)
        px, py = cx + int(size * 0.18), cy + int(size * 0.10)
        cons_mask = ((X - px) ** 2 / (size // 12) ** 2 + (Y - py) ** 2 / (size // 10) ** 2) < 1
        img[cons_mask] = 0.55 + rng2.random((size, size))[cons_mask] * 0.15

    elif pathology == "nodule":
        nx, ny = cx - int(size * 0.14), cy - int(size * 0.08)
        node_mask = (X - nx) ** 2 + (Y - ny) ** 2 < (size // 30) ** 2
        img[node_mask] = 0.70 + rng.random((size, size))[node_mask] * 0.2

    elif pathology == "pneumothorax":
        # Collapse right lung
        coll_lx = cx + int(size * 0.18)
        coll_mask = ((X - coll_lx) ** 2 / (size // 16) ** 2 + (Y - cy) ** 2 / (size // 22) ** 2) < 1
        img[coll_mask] = 0.35 + rng.random((size, size))[coll_mask] * 0.08
        # Air gap
        air_mask = (X > coll_lx + size // 16) & body_mask
        img[air_mask] = 0.02

    elif pathology == "covid":
        # Bilateral ground glass — both lungs slightly brighter
        for lx_off in [-int(size * 0.18), int(size * 0.18)]:
            lx = cx + lx_off
            ly = cy
            gg_mask = ((X - lx) ** 2 / (size // 16) ** 2 + (Y - ly) ** 2 / (size // 22) ** 2) < 1
            img[gg_mask] = np.clip(img[gg_mask] + 0.18, 0, 1)

    img = np.clip(img, 0, 1)
    arr = (img * 255).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (3, 3), 0.6)
    return Image.fromarray(arr).convert("RGB")


CT_CASES = [
    ("ct_chest_normal_01.jpg",        "normal",        1),
    ("ct_chest_normal_02.jpg",        "normal",        2),
    ("ct_chest_normal_03.jpg",        "normal",        3),
    ("ct_chest_consolidation_01.jpg", "consolidation", 4),
    ("ct_chest_consolidation_02.jpg", "consolidation", 5),
    ("ct_chest_nodule_01.jpg",        "nodule",        6),
    ("ct_chest_nodule_02.jpg",        "nodule",        7),
    ("ct_chest_pneumothorax_01.jpg",  "pneumothorax",  8),
    ("ct_chest_covid_01.jpg",         "covid",         9),
    ("ct_chest_covid_02.jpg",         "covid",         10),
]


def generate_ct():
    print("\n[2/3] Generating CT Chest Scan images...")
    for fname, pathology, seed in CT_CASES:
        dest = CT_DIR / fname
        if dest.exists():
            print(f"  skip  {fname}")
            continue
        img = make_ct_chest(fname, size=512, pathology=pathology, seed=seed)
        img.save(str(dest), quality=95)
        print(f"  [OK]  {fname}  ({pathology})")
    print(f"  CT: {len(list(CT_DIR.glob('*.jpg')))} images -> {CT_DIR}")


# =============================================================================
# 3.  ABDOMINAL ULTRASOUND  (B-mode appearance)
# =============================================================================

def make_ultrasound(
    organ: str = "liver",
    size: int = 512,
    pathology: str = "normal",
    seed: int = 0,
) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    # Ultrasound fan shape (sector probe)
    apex_x, apex_y = size // 2, int(size * 0.04)
    fan_angle = 70   # degrees total
    fan_depth  = int(size * 0.90)

    for y in range(size):
        for x in range(size):
            dx, dy = x - apex_x, y - apex_y
            if dy <= 0:
                continue
            angle = math.degrees(math.atan2(dx, dy))
            dist  = math.sqrt(dx ** 2 + dy ** 2)
            if abs(angle) < fan_angle / 2 and dist < fan_depth:
                # Base speckle noise (tissue background)
                img[y, x] = 0.22 + rng.random() * 0.12

    # Organ-specific texture
    if organ == "liver":
        # Liver parenchyma — homogeneous mid-echo
        liver_mask_y = slice(int(size * 0.18), int(size * 0.70))
        liver_mask_x = slice(int(size * 0.15), int(size * 0.82))
        texture = rng.random((size, size)) * 0.18 + 0.28
        img[liver_mask_y, liver_mask_x] = np.where(
            img[liver_mask_y, liver_mask_x] > 0,
            texture[liver_mask_y, liver_mask_x],
            img[liver_mask_y, liver_mask_x],
        )
        # Portal vein (dark tubular structure)
        cv2.line(img, (int(size * 0.30), int(size * 0.35)),
                 (int(size * 0.65), int(size * 0.55)), 0.05, 8)
        cv2.line(img, (int(size * 0.30), int(size * 0.35)),
                 (int(size * 0.65), int(size * 0.55)), 0.55, 2)

        if pathology == "fatty":
            img[liver_mask_y, liver_mask_x] += 0.12  # brighter = fatty infiltration
        elif pathology == "mass":
            mx, my = int(size * 0.50), int(size * 0.42)
            mass_mask = (np.ogrid[:size, :size][1] - mx) ** 2 + (np.ogrid[:size, :size][0] - my) ** 2 < (size // 12) ** 2
            img[mass_mask] = 0.12 + rng.random((size, size))[mass_mask] * 0.08
            # Bright halo
            halo = (np.ogrid[:size, :size][1] - mx) ** 2 + (np.ogrid[:size, :size][0] - my) ** 2
            halo_mask = (halo < (size // 10) ** 2) & (halo > (size // 12) ** 2)
            img[halo_mask] = 0.68

    elif organ == "kidney":
        # Kidney — elliptical with echogenic sinus
        ky, kx = int(size * 0.45), int(size * 0.48)
        krx, kry = int(size * 0.22), int(size * 0.14)
        Y, X = np.ogrid[:size, :size]
        kidney_mask = ((X - kx) ** 2 / krx ** 2 + (Y - ky) ** 2 / kry ** 2) < 1
        img[kidney_mask] = 0.20 + rng.random((size, size))[kidney_mask] * 0.10
        # Bright sinus
        sinus = ((X - kx) ** 2 / (krx // 2) ** 2 + (Y - ky) ** 2 / (kry // 2) ** 2) < 1
        img[sinus] = 0.65 + rng.random((size, size))[sinus] * 0.15
        if pathology == "stone":
            sx, sy = kx + krx // 3, ky - kry // 4
            stone = (X - sx) ** 2 + (Y - sy) ** 2 < (size // 40) ** 2
            img[stone] = 0.98   # bright stone
            # Acoustic shadow (dark stripe below stone)
            img[sy:sy + int(size * 0.3), sx - size // 40:sx + size // 40] = 0.01

    # Smooth speckle
    img = np.clip(img, 0, 1)
    arr = (img * 255).astype(np.uint8)
    arr = cv2.GaussianBlur(arr, (5, 5), 1.2)
    # Add fine speckle grain
    grain = rng.integers(0, 18, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) + grain - 9, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


US_CASES = [
    ("us_liver_normal_01.jpg",   "liver",  "normal", 1),
    ("us_liver_normal_02.jpg",   "liver",  "normal", 2),
    ("us_liver_fatty_01.jpg",    "liver",  "fatty",  3),
    ("us_liver_fatty_02.jpg",    "liver",  "fatty",  4),
    ("us_liver_mass_01.jpg",     "liver",  "mass",   5),
    ("us_liver_mass_02.jpg",     "liver",  "mass",   6),
    ("us_kidney_normal_01.jpg",  "kidney", "normal", 7),
    ("us_kidney_normal_02.jpg",  "kidney", "normal", 8),
    ("us_kidney_stone_01.jpg",   "kidney", "stone",  9),
    ("us_kidney_stone_02.jpg",   "kidney", "stone",  10),
]


def generate_ultrasound():
    print("\n[3/3] Generating Ultrasound images...")
    for fname, organ, pathology, seed in US_CASES:
        dest = US_DIR / fname
        if dest.exists():
            print(f"  skip  {fname}")
            continue
        img = make_ultrasound(organ=organ, size=512, pathology=pathology, seed=seed)
        img.save(str(dest), quality=95)
        print(f"  [OK]  {fname}  ({organ}, {pathology})")
    print(f"  Ultrasound: {len(list(US_DIR.glob('*.jpg')))} images -> {US_DIR}")


# =============================================================================
# 4.  Remaining audio files (patient 07 - 20)
# =============================================================================

REMAINING_AUDIO = [
    ("patient_07_pulmonary_embolism.wav",
     "48 year old female, 2 weeks post right knee replacement, sudden dyspnea and pleuritic chest pain. "
     "Heart rate 112, oxygen saturation 89 percent. D-dimer elevated. CT pulmonary angiography confirms bilateral emboli. "
     "Starting low molecular weight heparin. Considering thrombolysis."),
    ("patient_08_sepsis.wav",
     "68 year old male diabetic, confusion, fever 39.2, blood pressure 88 over 56, lactate 3.8. "
     "Sepsis secondary to urinary tract infection. Starting IV piperacillin tazobactam and fluid resuscitation."),
    ("patient_09_lung_cancer.wav",
     "63 year old male, 35 pack-year smoking history, 3 months worsening cough and haemoptysis, 6 kilogram weight loss. "
     "Chest X-ray shows 4 centimetre right upper lobe mass. CT confirms spiculated lesion with mediastinal lymphadenopathy. "
     "Urgent biopsy and thoracic oncology referral."),
    ("patient_10_atelectasis.wav",
     "55 year old female, day 2 post laparoscopic cholecystectomy, fever 38.1, oxygen saturation 93 percent. "
     "Chest X-ray shows left lower lobe atelectasis. Incentive spirometry hourly, physiotherapy referred."),
    ("patient_11_aortic_stenosis.wav",
     "74 year old male, exertional chest pain, syncope, progressive dyspnea. "
     "Echocardiogram shows severe aortic stenosis, valve area 0.7 centimetres squared, mean gradient 52. "
     "Referring for transcatheter aortic valve implantation."),
    ("patient_12_interstitial_lung_disease.wav",
     "61 year old female non-smoker, 8 months progressive dyspnea and dry cough, finger clubbing. "
     "High resolution CT shows bilateral basal honeycombing. Idiopathic pulmonary fibrosis. Starting nintedanib."),
    ("patient_13_acute_asthma.wav",
     "19 year old female, severe asthma attack, using accessory muscles, peak flow 35 percent of best. "
     "Giving salbutamol nebulisers, ipratropium, prednisolone, and magnesium sulphate intravenous. ICU review arranged."),
    ("patient_14_cardiac_tamponade.wav",
     "52 year old male with lung cancer, dyspnea and hypotension. Jugular venous distension. "
     "Echocardiogram confirms cardiac tamponade. Performing urgent pericardiocentesis. 750 millilitres drained."),
    ("patient_15_dvt_cellulitis.wav",
     "45 year old female, right leg pain, swelling, erythema. Doppler confirms right femoral deep vein thrombosis. "
     "Starting apixaban for DVT. Starting IV flucloxacillin for cellulitis."),
    ("patient_16_hypertensive_emergency.wav",
     "58 year old male, blood pressure 210 over 130, severe headache, blurred vision, papilloedema on fundoscopy. "
     "Hypertensive emergency. Starting IV labetalol infusion. MRI brain arranged."),
    ("patient_17_diabetic_ketoacidosis.wav",
     "22 year old type 1 diabetic, vomiting and abdominal pain. Blood glucose 32, pH 7.12, ketones 4 plus. "
     "Severe diabetic ketoacidosis. Starting saline, fixed rate insulin infusion, hourly monitoring."),
    ("patient_18_upper_gi_bleed.wav",
     "67 year old male on warfarin, haematemesis and melaena. Blood pressure 90 over 60, INR 3.8. "
     "Giving tranexamic acid, IV proton pump inhibitor, blood transfusion. Reversing warfarin. Urgent endoscopy."),
    ("patient_19_stroke.wav",
     "69 year old male, sudden left facial droop, arm weakness, dysarthria. Onset 1 hour 45 minutes ago. "
     "CT head: no haemorrhage. CT angiogram: right MCA occlusion. Giving IV alteplase. Transferring for thrombectomy."),
    ("patient_20_acute_kidney_injury.wav",
     "72 year old female on ramipril and ibuprofen, reduced urine output, creatinine 380 from baseline 90, "
     "potassium 6.1, peaked T waves. Acute kidney injury stage 3. Stopping nephrotoxins. Calcium gluconate, "
     "insulin dextrose. Nephrology review."),
]


def generate_remaining_audio():
    print(f"\n[4/4] Generating remaining {len(REMAINING_AUDIO)} audio files...")
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

    done = skipped = 0
    for fname, script in REMAINING_AUDIO:
        dest = AUDIO_DIR / fname
        if dest.exists():
            print(f"  skip  {fname}")
            skipped += 1
            continue
        engine.save_to_file(script, str(dest))
        engine.runAndWait()
        if dest.exists():
            print(f"  [OK]  {fname}")
            done += 1
        else:
            print(f"  [ERR] {fname} - pyttsx3 failed to save")

    total = len(list(AUDIO_DIR.glob("*.wav")))
    print(f"  Audio: {done} generated, {skipped} skipped | Total: {total} files")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Generating MRI / CT / Ultrasound samples + remaining audio")
    print("=" * 60)

    generate_mri()
    generate_ct()
    generate_ultrasound()
    generate_remaining_audio()

    print("\n" + "=" * 60)
    print("Done. Sample files:")
    print(f"  Brain MRI    : {len(list(MRI_DIR.glob('*.jpg')))} images  -> data/raw/mri_scans/")
    print(f"  CT Chest     : {len(list(CT_DIR.glob('*.jpg')))} images  -> data/raw/ct_scans/")
    print(f"  Ultrasound   : {len(list(US_DIR.glob('*.jpg')))} images  -> data/raw/ultrasound/")
    print(f"  Audio samples: {len(list(AUDIO_DIR.glob('*.wav')))} files   -> data/raw/audio_samples/")
    print("=" * 60)
