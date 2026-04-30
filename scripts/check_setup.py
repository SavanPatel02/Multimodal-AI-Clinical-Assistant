"""
Run this to verify your environment is ready before starting the API.

Usage:
    venv/Scripts/python.exe scripts/check_setup.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OK   = "[OK]  "
WARN = "[WARN]"
FAIL = "[FAIL]"

errors = 0

# 1. Python version
major, minor = sys.version_info[:2]
if (major, minor) >= (3, 10):
    print(f"{OK} Python {major}.{minor}")
else:
    print(f"{FAIL} Python {major}.{minor} — need 3.10+")
    errors += 1

# 2. Core imports
modules = [
    ("faster_whisper", "faster-whisper"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("cv2", "opencv-python-headless"),
    ("PIL", "Pillow"),
    ("ollama", "ollama"),
    ("langchain_community", "langchain-community"),
    ("faiss", "faiss-cpu"),
    ("sentence_transformers", "sentence-transformers"),
    ("spacy", "spacy"),
    ("fastapi", "fastapi"),
    ("streamlit", "streamlit"),
]
for mod, pkg in modules:
    try:
        __import__(mod)
        print(f"{OK} {pkg}")
    except ImportError:
        print(f"{FAIL} {pkg} — run: venv/Scripts/python.exe -m pip install {pkg}")
        errors += 1

# 3. spaCy model
try:
    import spacy
    spacy.load("en_core_web_sm")
    print(f"{OK} spaCy model en_core_web_sm")
except OSError:
    print(f"{FAIL} spaCy model missing — run: venv/Scripts/python.exe -m spacy download en_core_web_sm")
    errors += 1

# 4. Ollama running
try:
    import ollama
    client = ollama.Client(host="http://localhost:11434")
    models = client.list()
    names = [m.model for m in models.models]
    if names:
        print(f"{OK} Ollama running — models: {', '.join(names[:3])}")
        if any("mistral" in n or "phi" in n or "llama" in n for n in names):
            print(f"{OK} LLM model found")
        else:
            print(f"{WARN} No LLM model pulled yet — run: ollama pull mistral:7b-instruct")
    else:
        print(f"{WARN} Ollama running but no models pulled — run: ollama pull mistral:7b-instruct")
except Exception as e:
    print(f"{WARN} Ollama not reachable — start it with: ollama serve")
    print(f"       Then: ollama pull mistral:7b-instruct  (4.1 GB, fits in RTX 4060 8GB)")

# 5. FAISS index
from pathlib import Path
if Path("data/pubmed_index/faiss_index").exists():
    print(f"{OK} FAISS index exists")
else:
    print(f"{WARN} No FAISS index — RAG will return empty results.")
    print(f"       Build it: venv/Scripts/python.exe scripts/build_rag_index.py")

# 6. ffmpeg
import shutil
if shutil.which("ffmpeg"):
    print(f"{OK} ffmpeg found")
else:
    print(f"{WARN} ffmpeg not in PATH — MP3/M4A audio won't convert (WAV works fine)")

print()
if errors == 0:
    print("Environment ready. Start the API with:")
    print("  venv/Scripts/python.exe -m uvicorn src.api.main:app --reload")
    print("Start the UI with:")
    print("  venv/Scripts/python.exe -m streamlit run frontend/app.py")
else:
    print(f"{errors} error(s) found — fix them before starting.")
    sys.exit(1)
