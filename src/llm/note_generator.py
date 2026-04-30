from __future__ import annotations

import json
import os
import re
import time
from typing import Iterator, Optional

import ollama

from src.llm.prompts import SYSTEM_PROMPT, SOAP_GENERATION_TEMPLATE

# Model options for RTX 4060 8GB:
#   phi3:mini          → 2.3 GB VRAM, fast, solid reasoning
#   mistral:7b-instruct → 4.1 GB VRAM, better quality, recommended
#   llama3:8b-instruct  → 4.7 GB VRAM, slightly better, fits if nothing else on GPU
#
# Set via .env: OLLAMA_MODEL=mistral:7b-instruct
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MAX_TOKENS = 1024
TEMPERATURE = 0.2


class ClinicalNoteGenerator:
    """
    LLM wrapper using Ollama for local inference on GPU.
    Ollama manages VRAM automatically — the model stays loaded between requests.
    No transformers/bitsandbytes needed.
    """

    def __init__(self, model: str = DEFAULT_MODEL, host: str = OLLAMA_HOST):
        self.model = model
        self._client = ollama.Client(host=host)
        self._verified = False

    def load(self) -> None:
        """Verify Ollama is running and the model is available. Pull if not."""
        try:
            models = self._client.list()
            available = [m.model for m in models.models]
            if not any(self.model in m for m in available):
                print(f"Pulling {self.model} via Ollama (first-time download)...")
                self._client.pull(self.model)
            self._verified = True
        except Exception as e:
            raise RuntimeError(
                f"Ollama not reachable at {OLLAMA_HOST}.\n"
                "Start it with: ollama serve\n"
                f"Then pull the model: ollama pull {self.model}\n"
                f"Error: {e}"
            )

    def _build_messages(
        self,
        transcription: str,
        pathology_findings: str,
        rag_context: str,
        imaging_modality: str,
    ) -> list[dict]:
        user_content = SOAP_GENERATION_TEMPLATE.format(
            transcription=transcription,
            imaging_modality=imaging_modality,
            pathology_findings=pathology_findings,
            rag_context=rag_context,
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate(
        self,
        transcription: str,
        pathology_findings: str,
        rag_context: str,
        imaging_modality: str = "Chest X-Ray",
        max_tokens: int = MAX_TOKENS,
    ) -> dict:
        """
        Synchronous generation via Ollama.

        Returns:
            {
                "raw_text": str,
                "parsed": dict,
                "input_tokens": int,
                "output_tokens": int,
                "latency_s": float,
            }
        """
        if not self._verified:
            self.load()

        messages = self._build_messages(
            transcription, pathology_findings, rag_context, imaging_modality
        )

        t0 = time.perf_counter()
        response = self._client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": TEMPERATURE,
                "num_predict": max_tokens,
            },
        )
        latency = time.perf_counter() - t0

        raw_text = response.message.content
        parsed = self._extract_json(raw_text)

        return {
            "raw_text": raw_text,
            "parsed": parsed,
            "input_tokens": response.prompt_eval_count or 0,
            "output_tokens": response.eval_count or 0,
            "latency_s": round(latency, 3),
        }

    def stream(
        self,
        transcription: str,
        pathology_findings: str,
        rag_context: str,
        imaging_modality: str = "Chest X-Ray",
        max_tokens: int = MAX_TOKENS,
    ) -> Iterator[str]:
        """
        Streaming generator — yields tokens one by one.
        Use with FastAPI SSE via EventSourceResponse.
        """
        if not self._verified:
            self.load()

        messages = self._build_messages(
            transcription, pathology_findings, rag_context, imaging_modality
        )

        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            stream=True,
            options={
                "temperature": TEMPERATURE,
                "num_predict": max_tokens,
            },
        ):
            token = chunk.message.content
            if token:
                yield token

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract the first valid JSON object from LLM output."""
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"raw": text, "parse_error": True}


_generator: Optional[ClinicalNoteGenerator] = None


def get_note_generator(model: str = DEFAULT_MODEL) -> ClinicalNoteGenerator:
    """Return the singleton note generator."""
    global _generator
    if _generator is None:
        _generator = ClinicalNoteGenerator(model=model)
        _generator.load()
    return _generator
