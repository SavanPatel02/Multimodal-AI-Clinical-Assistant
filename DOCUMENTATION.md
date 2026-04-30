# Multimodal AI Clinical Assistant — Full Technical Documentation

---

## 1. Introduction

The Multimodal AI Clinical Assistant is a locally-running AI system designed to assist
doctors in generating structured clinical documentation. A physician speaks naturally
about a patient case and uploads a medical image. The system processes both inputs
through multiple AI models and produces a structured clinical note along with
diagnosis suggestions, drug interaction warnings, and supporting medical literature.

Everything runs on a consumer gaming GPU (NVIDIA RTX 4060 8GB). No data is sent to
any cloud service. No subscription or API key is required. All tools used are
free and open source.

---

## 2. Objectives

- Reduce the time a doctor spends writing clinical notes after a patient visit
- Extract structured information (symptoms, diagnoses, medications) automatically
- Highlight relevant regions of medical images using explainable AI (GradCAM)
- Warn about dangerous drug combinations before prescriptions are finalized
- Surface relevant published medical research for each case automatically
- Demonstrate how multiple AI modalities can be combined into one unified pipeline

---

## 3. Tools and Technologies

### 3.1 Speech-to-Text (Audio Transcription)

**Tool:** faster-whisper (small model)
**Full name:** Fast implementation of OpenAI Whisper using CTranslate2
**Runs on:** CPU (leaves full GPU for the language model)
**Model size:** 244 MB
**What it does:** Converts the doctor's spoken voice note into text.
Supports English and other languages. Includes VAD (Voice Activity Detection —
automatically skips silent parts of the recording).

### 3.2 Medical Image Analysis

**Tool:** DenseNet121 from torchvision + GradCAM
**Full name:** Densely Connected Convolutional Network, 121 layers
**Runs on:** CPU
**Model size:** ~30 MB (ImageNet pretrained weights)
**What it does:**
- Extracts a 1024-dimensional feature vector from the chest X-ray
- Scores 14 chest pathologies (Atelectasis, Cardiomegaly, Consolidation,
  Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule,
  Pleural Thickening, Pneumonia, Pneumothorax)
- Generates a GradCAM heatmap showing which regions of the image influenced
  the model's output (red = high attention, blue = low attention)

**GradCAM (Gradient-weighted Class Activation Mapping):**
A technique that uses gradients flowing back through the network to highlight
which spatial regions were most important for a specific prediction. Makes the
model's decision process visible and interpretable to clinicians.

### 3.3 Large Language Model

**Tool:** Mistral 7B Instruct via Ollama
**Full name:** Mistral 7 Billion Parameter Instruction-Tuned Language Model
**Runs on:** GPU (NVIDIA RTX 4060 8GB — uses ~4.1 GB VRAM)
**Model size:** 4.1 GB (Q4_K_M quantized — 4-bit quantization)
**What it does:** Takes the transcribed text, image findings, and retrieved
literature as input and generates a structured SOAP clinical note in JSON format.

**Ollama:** A tool that manages local AI models — handles downloading, loading
onto GPU, and serving via a simple HTTP API. No coding required to use it.

**SOAP Note:** A standard medical documentation format:
- S — Subjective (what the patient reports)
- O — Objective (measurable findings, vitals, imaging)
- A — Assessment (diagnosis and clinical reasoning)
- P — Plan (treatment, medications, follow-up)

**Q4_K_M Quantization:** A compression technique that reduces model precision
from 16-bit floating point to 4-bit integers. Reduces memory from ~14 GB to
~4.1 GB with minimal quality loss.

### 3.4 Retrieval-Augmented Generation (RAG)

**Tools:** FAISS + sentence-transformers + LangChain + PubMed Entrez API
**Full names:**
- FAISS: Facebook AI Similarity Search
- RAG: Retrieval-Augmented Generation
- NCBI Entrez API: National Center for Biotechnology Information Entrez API

**What it does:**
When the doctor mentions symptoms (e.g., "productive cough, consolidation"),
the system converts those words into a numerical vector and searches a local
database of PubMed medical abstracts for the most similar content.
The top 3 most relevant abstracts are then given to the language model as
additional context before it writes the SOAP note.

This improves the accuracy and clinical grounding of the generated notes
because the model can reference real published research.

**Embedding model:** all-MiniLM-L6-v2 (sentence-transformers)
Converts text into 384-dimensional vectors that capture semantic meaning.
Similar clinical concepts end up close together in this vector space.

**Note on PubMed access:** Only abstracts (summaries) are freely available
through the NCBI Entrez API. Full-text papers often require institutional
journal subscriptions.

### 3.5 Named Entity Recognition (NER)

**Tool:** spaCy with custom clinical rule patterns
**Full name:** Named Entity Recognition
**Runs on:** CPU
**What it does:** Reads the generated SOAP note and identifies and labels:
- Symptoms (cough, fever, dyspnea, chest pain, etc.)
- Diagnoses (pneumonia, heart failure, COPD, etc.)
- Medications (amoxicillin, furosemide, warfarin, etc.)
- Dosages (875 mg, 40 mg daily, etc.)
- Anatomy (right lower lobe, bilateral, left lung, etc.)
- Procedures (chest X-ray, echocardiogram, spirometry, etc.)

Results are displayed as colored highlighted text in the UI.

### 3.6 Drug Interaction Checker

**Source:** Curated clinical pharmacology database (25 interactions) + OpenFDA API
**Full name:** OpenFDA: Open Food and Drug Administration drug label database
**What it does:** After NER identifies medications in the note, each pair of
medications is checked against the drug interaction database.
Interactions are classified as:
- Major: potentially life-threatening, avoid combination
- Moderate: clinically significant, requires monitoring
- Minor: low clinical significance, patient counselling needed

### 3.7 Backend API

**Tool:** FastAPI
**Full name:** Fast Application Programming Interface framework
**What it does:** Provides HTTP endpoints that the Streamlit UI calls.
Runs all models asynchronously so the server can handle requests without freezing.
Supports SSE (Server-Sent Events) for streaming the LLM output token by token.

**SSE: Server-Sent Events** — A protocol where the server pushes data to the
browser continuously as it becomes available, rather than waiting for the full
response. Used for showing LLM tokens as they are generated in real-time.

**Pydantic:** A Python library used to define strict data structures for API
inputs and outputs. Validates all data automatically.

### 3.8 Frontend UI

**Tool:** Streamlit
**What it does:** Provides a web-based dashboard with:
- Audio file upload and waveform visualization
- Image upload and display
- Analysis results across 5 tabs
- Plotly interactive charts for pathology scores
- Color-coded entity highlighting

### 3.9 Testing

**Tool:** pytest
**What it does:** Automated tests for every module. 32 tests covering:
- Audio loading, resampling, waveform stats
- Image loading, CLAHE preprocessing, bounding boxes
- LLM prompt formatting, JSON extraction
- RAG index building and retrieval
- Full pipeline integration

---

## 4. System Architecture

```
                    DOCTOR INPUT
                         |
          +--------------+--------------+
          |                             |
    Voice Note (WAV)          Chest X-Ray (PNG/JPG)
          |                             |
          v                             v
  +---------------+           +------------------+
  | faster-whisper |           |  DenseNet121 CNN  |
  |  (CPU, small) |           |  (CPU, torchvision)|
  +-------+-------+           +--------+---------+
          |                            |
          | Transcription text         | 1024-dim feature vector
          |                            | 14 pathology scores
          |                            | GradCAM heatmap
          |                            |
          +------------+---------------+
                       |
                       v
              +----------------+
              | Prompt Builder  |
              | (fusion layer)  |
              +-------+--------+
                      |
                      | Query for relevant papers
                      v
              +----------------+
              | FAISS + ST     |
              | PubMed RAG     |
              | (CPU, 384-dim) |
              +-------+--------+
                      |
                      | Top-3 PubMed abstracts
                      v
              +----------------+
              | Mistral 7B     |
              | Instruct       |
              | (GPU, Ollama)  |
              | Q4_K_M 4.1GB  |
              +-------+--------+
                      |
                      | Structured SOAP JSON
                      v
              +----------------+
              | spaCy NER      |
              | Clinical Rules |
              | (CPU)          |
              +-------+--------+
                      |
                      | Entities + Drug Warnings
                      v
              +---------------------------+
              | ClinicalAssistantResponse |
              | (Pydantic JSON object)    |
              +---------------------------+
                      |
                      v
              +------------------+
              | FastAPI Backend  |
              | localhost:8000   |
              +--------+---------+
                       |
                       v
              +------------------+
              | Streamlit UI     |
              | localhost:8501   |
              +------------------+
```

---

## 5. Pipeline Flow (Step by Step)

**Step 1 — Audio Transcription**
- Input: WAV/MP3 audio file from the doctor
- Processing: faster-whisper loads the audio at 16kHz mono, runs the small
  Whisper model with VAD filtering, and returns a text transcription with
  word-level timestamps
- Output: Plain English transcription text

**Step 2 — Image Encoding**
- Input: Chest X-ray PNG/JPG image
- Processing: Image is converted to grayscale, CLAHE (Contrast Limited
  Adaptive Histogram Equalization) enhancement is applied, then it is
  resized to 224x224 pixels and normalized with ImageNet statistics.
  DenseNet121 extracts features, sigmoid activation converts logits to
  probabilities. GradCAM computes gradients for the top pathology.
- Output: 1024-dim feature vector, 14 pathology confidence scores,
  GradCAM heatmap image

**CLAHE: Contrast Limited Adaptive Histogram Equalization**
An image processing technique that improves local contrast in medical images.
Particularly useful for chest X-rays where regions of interest may be subtle.

**Step 3 — RAG Retrieval**
- Input: Concatenation of transcription text + top pathology names
- Processing: sentence-transformers encodes the query into a 384-dim vector.
  FAISS performs cosine similarity search against the PubMed abstract index.
  Top 3 results are retrieved and formatted.
- Output: 3 relevant PubMed abstract texts

**Step 4 — SOAP Note Generation**
- Input: Transcription + pathology findings + RAG abstracts (combined prompt)
- Processing: Mistral 7B receives a structured prompt asking for a SOAP note
  in JSON format. The model generates text token by token on the RTX 4060 GPU.
- Output: JSON with soap_note, primary_diagnosis, differential_diagnoses,
  recommended_medications, follow_up, confidence level

**Step 5 — Named Entity Recognition**
- Input: Raw LLM output text
- Processing: The JSON is parsed to extract clean SOAP text.
  spaCy with clinical rule patterns identifies medical entities.
  Entity spans are recorded with character positions for highlighting.
- Output: Bucketed entities (symptoms, diagnoses, medications, dosages,
  anatomy, procedures) + highlighted HTML

**Step 6 — Drug Interaction Check**
- Input: List of medications extracted by NER
- Processing: All medication pairs are checked against the local database.
  Interactions are sorted by severity (major > moderate > minor).
- Output: List of warnings with severity, description, and recommendation

**Step 7 — Response Assembly**
- All outputs are combined into a ClinicalAssistantResponse Pydantic object
- Telemetry is recorded: audio duration, image shape, token counts, latency
- Response is serialized to JSON and returned to the Streamlit frontend

---

## 6. API Reference

### GET /api/v1/health
Check if all models are loaded and ready.

Response:
```json
{
  "status": "ok",
  "models_loaded": true,
  "version": "1.0.0"
}
```

### POST /api/v1/analyze
Full synchronous inference. Returns complete JSON response.

Form fields:
- `audio` — Audio file (WAV, MP3, M4A, FLAC, OGG)
- `image` — Image file (JPG, PNG, BMP, TIFF)
- `language` — Audio language code, default: "en"
- `imaging_modality` — e.g. "Chest X-Ray", default: "Chest X-Ray"

Response fields:

| Field | Type | Description |
|-------|------|-------------|
| transcription | string | Whisper transcription of audio |
| soap_note | object | S/O/A/P sections |
| primary_diagnosis | string | Most likely diagnosis |
| differential_diagnoses | list | Alternative diagnoses |
| image_findings_summary | string | Radiology summary paragraph |
| recommended_medications | list | Suggested drugs |
| follow_up | string | Follow-up instructions |
| confidence | string | high / medium / low |
| gradcam_overlay_b64 | string | Base64 PNG of GradCAM heatmap |
| named_entities | object | symptoms, diagnoses, medications, etc. |
| drug_interactions | list | Warnings with severity |
| pathology_scores | object | 14 pathology probabilities |
| top_findings | list | Pathologies above 0.3 threshold |
| literature_references | list | Retrieved PubMed abstracts |
| audio_duration_s | float | Length of audio in seconds |
| image_shape | list | [height, width] of input image |
| input_tokens | int | Tokens fed to LLM |
| output_tokens | int | Tokens generated by LLM |
| total_latency_s | float | End-to-end time in seconds |

### POST /api/v1/analyze/stream
Streaming SSE version. Emits events progressively:

| Event | Data | When |
|-------|------|------|
| progress | {"step": "transcribing"} | Pipeline step started |
| transcription | {"text": "..."} | Whisper complete |
| vision | {"top_findings": [...], "pathology_scores": {...}} | DenseNet complete |
| token | {"token": "..."} | Each LLM token |
| done | Full result JSON | Everything complete |
| error | {"detail": "..."} | On any failure |

---

## 7. Datasets

### Montgomery County Chest X-Ray Dataset
- Source: US National Library of Medicine (NLM)
- URL: data.lhncbc.nlm.nih.gov
- Content: 138 chest X-rays from a TB screening program in Maryland, USA
- Format: PNG, high resolution (4000x4892 pixels typical)
- License: Public domain, no registration required
- Usage in this project: Testing the vision encoder

### PubMed Abstracts
- Source: NCBI Entrez API
- Content: Abstracts (summaries) of peer-reviewed medical papers
- License: Free for non-commercial use, abstracts are public domain
- Note: Full-text papers require journal subscriptions
- Usage in this project: RAG knowledge base for grounding LLM output

### Drug Interaction Database
- Source: Curated from clinical pharmacology literature + OpenFDA API
- Content: 25 clinically significant drug-drug interactions
- License: Public domain medical knowledge
- Usage in this project: Safety checking of prescribed medications

### Synthetic Audio Samples
- Source: Generated locally using pyttsx3 (Windows SAPI5)
- Content: 4 doctor voice notes covering pneumonia, heart failure, COPD, TB
- License: Generated by the user, no external rights
- Usage in this project: Testing and demonstration

---

## 8. Sample Outputs

### SOAP Note Example (Patient 01 — Pneumonia)

**Transcription (from Whisper):**
> Patient is a 58-year-old male presenting with productive cough for 5 days,
> fever of 38.9 degrees Celsius, and right-sided pleuritic chest pain.
> Chest X-Ray shows right lower lobe consolidation.
> Plan to start amoxicillin-clavulanate 875 milligrams twice daily for 7 days.

**SOAP Note (from Mistral 7B):**

Subjective:
> 58-year-old male presenting with productive cough for five days, fever of
> 38.9 degrees Celsius, and right-sided pleuritic chest pain. Reports
> shortness of breath on exertion. Past medical history of hypertension
> and type 2 diabetes.

Objective:
> Imaging modality: Chest X-Ray. CheXNet detected Consolidation: 47.5%,
> Pneumonia: 34.2%. Decreased breath sounds at the right base with
> dullness to percussion.

Assessment:
> Primary differential diagnosis is community-acquired pneumonia given the
> patient's symptoms and chest X-ray findings. Other possibilities include
> interstitial pneumonia due to atypical organisms such as Mycoplasma.

Plan:
> Start amoxicillin clavulanate 875 mg twice daily for seven days with
> follow-up in one week.

**Extracted Entities:**
- Symptoms: productive cough, fever, chest pain, shortness of breath
- Diagnoses: pneumonia, consolidation, community-acquired pneumonia
- Medications: amoxicillin clavulanate
- Dosages: 875 mg
- Anatomy: right lower lobe, right base

### GradCAM Output

The heatmap shows the model's attention on the chest X-ray.
Red areas indicate high attention. Blue indicates low attention.
In pneumonia cases, attention concentrates on the lower lobe consolidation region.

---

## 9. Known Limitations

**Vision Model Accuracy**
The DenseNet121 classifier head uses random initialization because no
publicly free pre-trained CheXNet weights are available without PhysioNet
registration. Pathology scores reflect ImageNet features, not true medical
diagnosis. The GradCAM heatmap is still meaningful — it shows what the
network finds structurally interesting.

**RAG Index Size**
The current FAISS index contains 15 curated abstracts. For production use,
index thousands of PubMed abstracts using the build_rag_index.py script
with your NCBI Entrez email.

**Drug Interaction Database**
The local database contains 25 interactions. For comprehensive coverage,
integrate the full DrugBank or SIDER database (requires free registration).

**No Medical Validation**
This system is for research and demonstration only. It has not been
clinically validated. All outputs must be reviewed by qualified medical
professionals before any clinical use. The system can hallucinate or
produce incorrect information.

**Language**
Optimized for English medical terminology. Other languages supported by
Whisper for transcription but NER patterns are English-only.

---

## 10. Glossary

| Term | Full Form / Meaning |
|------|---------------------|
| SOAP | Subjective, Objective, Assessment, Plan — standard clinical note format |
| NER | Named Entity Recognition — AI identifying specific types of words |
| RAG | Retrieval-Augmented Generation — using a search engine to give AI context |
| LLM | Large Language Model — AI model trained on text (e.g. Mistral 7B) |
| CNN | Convolutional Neural Network — AI model for image processing |
| GradCAM | Gradient-weighted Class Activation Mapping — AI explainability technique |
| FAISS | Facebook AI Similarity Search — fast vector similarity search library |
| VRAM | Video Random Access Memory — GPU memory |
| SSE | Server-Sent Events — streaming protocol for real-time data |
| API | Application Programming Interface — software communication layer |
| CLAHE | Contrast Limited Adaptive Histogram Equalization — image enhancement |
| VAD | Voice Activity Detection — detecting when speech is present |
| SOAP | Subjective Objective Assessment Plan — clinical note format |
| CAP | Community-Acquired Pneumonia |
| COPD | Chronic Obstructive Pulmonary Disease |
| HFrEF | Heart Failure with Reduced Ejection Fraction |
| TB | Tuberculosis |
| EF | Ejection Fraction — percentage of blood pumped by the heart per beat |
| BNP | B-type Natriuretic Peptide — blood marker for heart failure |
| CXR | Chest X-Ray |
| PA | Posteroanterior — standard chest X-ray view (X-ray taken from back to front) |
| NLM | National Library of Medicine (US) |
| NCBI | National Center for Biotechnology Information |
| FDA | Food and Drug Administration (US) |
| INR | International Normalised Ratio — warfarin monitoring blood test |
| RIPE | Rifampicin, Isoniazid, Pyrazinamide, Ethambutol — standard TB treatment |
| AFB | Acid-Fast Bacilli — TB bacteria detected by smear test |
| PCR | Polymerase Chain Reaction — DNA-based diagnostic test |
| CT | Computed Tomography — cross-sectional X-ray scan |
| HRCT | High-Resolution Computed Tomography |
| MRI | Magnetic Resonance Imaging |
| ICU | Intensive Care Unit |
| IV | Intravenous — medication given directly into a vein |
| SpO2 | Oxygen saturation measured by pulse oximeter |
| FBC | Full Blood Count — standard blood test panel |
| TID | Three Times a Day (medication frequency) |
| BID | Twice a Day (medication frequency) |
