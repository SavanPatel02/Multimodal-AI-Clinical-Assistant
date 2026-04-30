# Multimodal AI Clinical Assistant

A local, fully free AI system that helps doctors generate structured clinical notes.
A doctor speaks about a patient and uploads a medical image — the system produces
a complete SOAP note with diagnosis suggestions, drug interaction warnings, and
relevant medical literature — all running on a standard consumer GPU.

---

## What It Does

1. Doctor records a voice note describing a patient
2. Doctor uploads a chest X-ray or MRI image
3. The system transcribes the speech, analyzes the image, searches medical literature,
   and generates a structured clinical SOAP note using a local AI model
4. The output includes extracted symptoms, medications, diagnoses, drug warnings,
   and a GradCAM heatmap showing exactly where the AI looked on the X-ray

---

## System Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA RTX 4060 8GB (or any 8GB+ VRAM GPU) |
| RAM | 16 GB |
| Storage | 15 GB free (models + data) |
| OS | Windows 10/11 |
| Python | 3.10+ |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Speech-to-Text | faster-whisper (small model, CPU) |
| Medical Image Analysis | DenseNet121 + GradCAM (CPU, torchvision) |
| Language Model | Mistral 7B Instruct via Ollama (GPU) |
| Medical Literature | FAISS + sentence-transformers + PubMed abstracts |
| Named Entity Recognition | spaCy with clinical vocabulary rules |
| Backend API | FastAPI with async endpoints + SSE streaming |
| Frontend UI | Streamlit |
| Drug Interactions | Local JSON database (25 clinical interactions) |

---

## Project Structure

```
clinical-assistant/
├── data/
│   ├── raw/
│   │   ├── audio_samples/          # 4 synthetic doctor voice notes (WAV)
│   │   └── chest_xrays/            # 7 real chest X-rays (Montgomery County, NLM)
│   ├── processed/
│   │   └── drug_interactions.json  # 25 curated drug interaction rules
│   └── pubmed_index/
│       └── faiss_index/            # FAISS vector index of PubMed abstracts
├── frontend/
│   └── app.py                      # Streamlit dashboard
├── models/
│   ├── llm/                        # Fine-tuned LLM weights (optional)
│   ├── vision_encoder/             # Custom classifier weights (optional)
│   ├── whisper/                    # Cached Whisper weights
│   └── ner/                        # Custom NER model (optional)
├── notebooks/
│   ├── 01_whisper_experiments.ipynb
│   ├── 02_vision_encoder.ipynb
│   ├── 03_llm_finetuning.ipynb
│   └── 04_rag_pipeline.ipynb
├── scripts/
│   ├── check_setup.py              # Verify environment before starting
│   ├── download_datasets.py        # Download all free datasets
│   └── build_rag_index.py          # Build PubMed FAISS index
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app startup
│   │   ├── routes.py               # API endpoints
│   │   └── schemas.py              # Pydantic request/response models
│   ├── audio/
│   │   ├── transcriber.py          # faster-whisper wrapper
│   │   └── audio_utils.py          # Audio loading and processing
│   ├── fusion/
│   │   └── multimodal_pipeline.py  # Combines all modalities
│   ├── llm/
│   │   ├── note_generator.py       # Ollama LLM wrapper
│   │   ├── rag_pipeline.py         # FAISS retrieval pipeline
│   │   └── prompts.py              # All LLM prompt templates
│   ├── ner/
│   │   ├── entity_extractor.py     # spaCy NER + rule patterns
│   │   └── ner_utils.py            # Drug interaction checker
│   └── vision/
│       ├── image_encoder.py        # DenseNet121 + GradCAM
│       └── image_utils.py          # Image loading and preprocessing
├── tests/
│   ├── test_audio.py
│   ├── test_vision.py
│   ├── test_llm.py
│   └── test_pipeline.py
├── .env.example                    # Environment variable template
├── .gitignore
├── pytest.ini
├── requirements.txt
├── STARTUP.md                      # How to start the project
└── DOCUMENTATION.md                # Full technical documentation
```

---

## Quick Start

See [STARTUP.md](STARTUP.md) for complete step-by-step instructions.

```bash
# 1. Install dependencies
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Download datasets
python scripts/download_datasets.py

# 3. Start Ollama (separate terminal)
ollama serve
ollama pull mistral:7b-instruct

# 4. Start API
uvicorn src.api.main:app --reload

# 5. Start UI
streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

---

## Sample Output

Upload `data/raw/audio_samples/patient_01_pneumonia.wav` and
`data/raw/chest_xrays/MCUCXR_0001_0.png` to see a full example output including:

- Transcribed doctor note
- SOAP note (Subjective / Objective / Assessment / Plan)
- Primary diagnosis with differentials
- GradCAM heatmap on the X-ray
- Extracted entities (symptoms, medications, dosages)
- Drug interaction warnings
- PubMed literature references

---

## Datasets Used (All Free)

| Dataset | Source | License |
|---------|--------|---------|
| Montgomery County CXR | US National Library of Medicine | Public Domain |
| PubMed Abstracts | NCBI Entrez API | Public Domain |
| Drug Interactions | Curated + OpenFDA API | Public Domain |
| Audio Samples | Generated with pyttsx3 | Generated locally |

---

## Running Tests

```bash
venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: 32 passed.
