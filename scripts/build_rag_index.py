"""
Build the PubMed RAG index using free Entrez API (abstracts only, no paywall).
Run once before starting the API.

Usage:
    venv/Scripts/python.exe scripts/build_rag_index.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.llm.rag_pipeline import PubMedRAGPipeline

# Free search queries — PubMed abstracts only, no payment needed
QUERIES = [
    "community acquired pneumonia chest X-ray treatment guidelines",
    "pulmonary edema diagnosis management echocardiography",
    "COPD exacerbation treatment spirometry",
    "tuberculosis radiology findings diagnosis",
    "pleural effusion causes diagnosis treatment",
    "lung nodule malignancy risk Fleischner",
    "atelectasis chest X-ray causes",
    "pneumothorax diagnosis management",
    "cardiac tamponade pericardial effusion",
    "consolidation chest radiograph differential diagnosis",
]

RESULTS_PER_QUERY = 15   # 15 × 10 queries = 150 free abstracts total
EMAIL = os.getenv("ENTREZ_EMAIL", "researcher@example.com")

if EMAIL == "researcher@example.com":
    print("WARNING: Set ENTREZ_EMAIL env var to your real email (NCBI requirement).")
    print("         Using placeholder — NCBI may throttle requests.\n")

print(f"Fetching PubMed abstracts ({len(QUERIES)} queries × {RESULTS_PER_QUERY} results)...")
print("Note: Only abstracts are free. Full-text papers require institutional access.\n")

rag = PubMedRAGPipeline()

try:
    rag.fetch_and_build(QUERIES, max_results=len(QUERIES) * RESULTS_PER_QUERY)
    print("\nFAISS index built and saved to data/pubmed_index/faiss_index/")
    print("You can now start the API server.")
except Exception as e:
    print(f"\nError: {e}")
    print("\nIf Entrez API fails, build a minimal index from local data:")
    print("  from src.llm.rag_pipeline import PubMedRAGPipeline")
    print("  rag = PubMedRAGPipeline()")
    print("  rag.build_index([{'pmid':'1','title':'Test','abstract':'Test abstract.'}])")
    sys.exit(1)
