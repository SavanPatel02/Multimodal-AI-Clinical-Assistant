# How to Start the Clinical AI Assistant

Follow these steps every time you open the project after closing everything.

---

## Before You Begin — One-Time Setup

These steps only need to be done ONCE on a fresh machine.
If you have already done them, skip to the "Every Time" section below.

**Step 1 — Install Python 3.10**
Download from https://www.python.org/downloads/ and install.

**Step 2 — Install Ollama**
Download from https://ollama.com/download/windows and install.
Ollama starts automatically after install.

**Step 3 — Install FFmpeg**
Open PowerShell and run:
```powershell
winget install Gyan.FFmpeg
```
Then close and reopen PowerShell so FFmpeg is in PATH.

**Step 4 — Create virtual environment and install packages**
```powershell
cd "d:\Savan Project\Multimodal AI clinical assistant\clinical-assistant"
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m spacy download en_core_web_sm
```

**Step 5 — Download datasets**
```powershell
venv\Scripts\python.exe scripts/download_datasets.py
```

**Step 6 — Pull the AI model (4.1 GB download, once)**
```powershell
ollama pull mistral:7b-instruct
```

---

## Every Time You Start the Project

You need 3 terminals open at the same time.

### Terminal 1 — Check everything is ready
```powershell
cd "d:\Savan Project\Multimodal AI clinical assistant\clinical-assistant"
venv\Scripts\python.exe scripts/check_setup.py
```
All items should show [OK]. Fix any [FAIL] items before continuing.

---

### Terminal 2 — Start the API server
```powershell
cd "d:\Savan Project\Multimodal AI clinical assistant\clinical-assistant"
venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Wait until you see this line:
```
INFO:     Application startup complete.
```

This loads all models into memory. It takes about 30-60 seconds on first run.
Ollama loads Mistral 7B onto your RTX 4060 GPU automatically.

---

### Terminal 3 — Start the UI
```powershell
cd "d:\Savan Project\Multimodal AI clinical assistant\clinical-assistant"
venv\Scripts\python.exe -m streamlit run frontend\app.py
```

Your browser opens automatically at http://localhost:8501

---

## Using the Application

1. Open http://localhost:8501 in your browser
2. Upload a voice note (WAV file) — use files from `data/raw/audio_samples/`
3. Upload a chest X-ray (PNG/JPG) — use files from `data/raw/chest_xrays/`
4. Click **Analyze and Generate SOAP Note**
5. Wait 20-60 seconds (Mistral 7B is generating the note on your GPU)
6. Browse the tabs: SOAP Note / Image Findings / Entities / Literature / Raw JSON

---

## Sample Files to Test With

| Audio File | Condition |
|------------|-----------|
| `data/raw/audio_samples/patient_01_pneumonia.wav` | Community-acquired pneumonia |
| `data/raw/audio_samples/patient_02_heart_failure.wav` | Heart failure with reduced EF |
| `data/raw/audio_samples/patient_03_copd.wav` | COPD exacerbation |
| `data/raw/audio_samples/patient_04_tuberculosis.wav` | Pulmonary tuberculosis |

| X-Ray File | Notes |
|------------|-------|
| `data/raw/chest_xrays/MCUCXR_0001_0.png` | Normal PA chest X-ray |
| `data/raw/chest_xrays/MCUCXR_0002_0.png` | Normal PA chest X-ray |
| Any from `data/raw/chest_xrays/` | All are real Montgomery County CXRs |

---

## Stopping the Application

- Press `Ctrl + C` in Terminal 2 (API) to stop it
- Press `Ctrl + C` in Terminal 3 (Streamlit) to stop it
- Ollama keeps running in the background (that is fine)

---

## Troubleshooting

**"Cannot connect to API" in the browser**
- Terminal 2 (uvicorn) is not running or crashed
- Start Terminal 2 again and wait for "Application startup complete"

**"ollama: command not found"**
- Close PowerShell and open a new one
- Or run: `& "C:\Users\savan\AppData\Local\Programs\Ollama\ollama.exe" serve`

**API starts but crashes immediately**
- Check that Ollama is running: open http://localhost:11434 in browser
- Should show "Ollama is running"
- If not, Ollama crashed — restart it from Start Menu or run `ollama serve`

**Analysis takes very long (more than 3 minutes)**
- Mistral 7B is running on GPU — normal time is 20-60 seconds
- If it takes longer, check Task Manager — GPU should be at 80-100% usage
- If GPU usage is 0%, Ollama may not have loaded the model onto the GPU

**Port already in use error**
```powershell
# Find and kill the process using port 8000
netstat -ano | findstr :8000
taskkill /PID <number shown> /F
```

**Rebuild the PubMed index**
```powershell
# If RAG returns no results or index is corrupted:
venv\Scripts\python.exe scripts/build_rag_index.py
```

---

## API Endpoints (for developers)

| Endpoint | Method | Description |
|----------|--------|-------------|
| http://localhost:8000/ | GET | API welcome message |
| http://localhost:8000/docs | GET | Interactive API documentation |
| http://localhost:8000/api/v1/health | GET | Check if all models are loaded |
| http://localhost:8000/api/v1/analyze | POST | Full inference (returns JSON) |
| http://localhost:8000/api/v1/analyze/stream | POST | Streaming inference (SSE) |
