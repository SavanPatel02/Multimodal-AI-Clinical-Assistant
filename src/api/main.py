from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.fusion.multimodal_pipeline import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
RAG_INDEX_PATH = os.getenv("RAG_INDEX_PATH", "data/pubmed_index/faiss_index")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all components once at startup. Ollama must already be running."""
    log.info("Starting up — loading all components...")
    log.info("  LLM  : %s  (GPU via Ollama)", OLLAMA_MODEL)
    log.info("  ASR  : faster-whisper %s  (CPU)", WHISPER_MODEL)
    log.info("  Vision: DenseNet121  (CPU)")
    log.info("  RAG  : FAISS + sentence-transformers  (CPU)")
    log.info("  NER  : spaCy  (CPU)")

    pipeline = get_pipeline(
        whisper_model=WHISPER_MODEL,
        ollama_model=OLLAMA_MODEL,
        rag_index_path=RAG_INDEX_PATH,
    )
    app.state.pipeline = pipeline
    log.info("All components ready. API is up.")
    yield
    log.info("Shutdown.")


app = FastAPI(
    title="Multimodal AI Clinical Assistant",
    description=(
        "Speech (Whisper CPU) + Medical Image (DenseNet CPU) + "
        "SOAP Note (Mistral via Ollama GPU) + RAG (PubMed abstracts) + Clinical NER."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {
        "message": "Multimodal AI Clinical Assistant API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
