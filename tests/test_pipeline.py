"""Integration tests for the full multimodal pipeline."""
from __future__ import annotations

import json
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.fusion.multimodal_pipeline import (
    ClinicalAssistantResponse,
    MultimodalClinicalPipeline,
    SOAPNote,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_wav(path: Path, duration_s: float = 1.0, sr: int = 16000) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    samples = (np.sin(2 * np.pi * 440 * t) * 0.5 * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())


def _make_jpg(path: Path) -> None:
    arr = np.random.randint(0, 200, (224, 224, 3), dtype=np.uint8)
    Image.fromarray(arr).save(str(path), format="JPEG")


@pytest.fixture
def sample_files(tmp_path):
    audio = tmp_path / "test.wav"
    image = tmp_path / "test.jpg"
    _make_wav(audio)
    _make_jpg(image)
    return audio, image


MOCK_SOAP_JSON = json.dumps({
    "soap_note": {
        "subjective": "Patient has cough and fever.",
        "objective": "Consolidation right lower lobe.",
        "assessment": "Community-acquired pneumonia.",
        "plan": "Amoxicillin 500mg TID × 7 days.",
    },
    "primary_diagnosis": "Community-Acquired Pneumonia",
    "differential_diagnoses": ["Pulmonary TB", "COVID-19 Pneumonia"],
    "image_findings_summary": "Right lower lobe consolidation noted.",
    "recommended_medications": ["amoxicillin", "ibuprofen"],
    "follow_up": "Return in 7 days or sooner if worsening.",
    "confidence": "high",
    "literature_references": [],
})


# ── Pipeline integration tests ────────────────────────────────────────────────

class TestMultimodalPipeline:
    def _make_pipeline(self):
        pipeline = MultimodalClinicalPipeline()

        # Mock transcriber
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = {
            "text": "Patient has cough and fever for 3 days.",
            "segments": [],
            "language": "en",
            "duration_s": 1.0,
            "latency_s": 0.5,
            "waveform_stats": {"rms": 0.1, "peak": 0.5, "duration_samples": 16000, "is_clipped": False},
        }

        # Mock image encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = {
            "feature_vector": [0.0] * 1024,
            "pathology_scores": {"Pneumonia": 0.85, "Atelectasis": 0.12},
            "top_findings": ["Pneumonia"],
            "image_metadata": {"height": 224, "width": 224, "mode": "RGB", "shape": (224, 224, 3)},
            "latency_s": 0.1,
        }

        # Mock LLM generator
        mock_generator = MagicMock()
        mock_generator.generate.return_value = {
            "raw_text": MOCK_SOAP_JSON,
            "parsed": json.loads(MOCK_SOAP_JSON),
            "input_tokens": 512,
            "output_tokens": 256,
            "latency_s": 1.5,
        }

        # Mock RAG
        mock_rag = MagicMock()
        mock_rag.retrieve.return_value = [
            {"title": "Pneumonia Study", "pmid": "99999", "text": "CAP management...", "score": 0.9}
        ]
        mock_rag.format_context.return_value = "[1] Pneumonia Study\nPMID: 99999\nCAP management..."

        # Mock NER
        mock_ner = MagicMock()
        mock_ner.extract.return_value = {
            "symptoms": ["cough", "fever"],
            "diagnoses": ["pneumonia"],
            "medications": ["amoxicillin", "ibuprofen"],
            "dosages": ["500mg"],
            "procedures": [],
            "anatomy": ["right lower lobe"],
            "drug_interactions": [],
            "entity_spans": [],
        }
        mock_ner.highlight_html.return_value = "<span>highlighted text</span>"

        pipeline._transcriber = mock_transcriber
        pipeline._encoder = mock_encoder
        pipeline._generator = mock_generator
        pipeline._rag = mock_rag
        pipeline._ner = mock_ner

        return pipeline

    def test_run_returns_clinical_response(self, sample_files):
        audio_path, image_path = sample_files
        pipeline = self._make_pipeline()
        result = pipeline.run(audio_path=audio_path, image_path=image_path)

        assert isinstance(result, ClinicalAssistantResponse)
        assert result.transcription == "Patient has cough and fever for 3 days."
        assert isinstance(result.soap_note, SOAPNote)
        assert result.soap_note.subjective != ""
        assert result.primary_diagnosis == "Community-Acquired Pneumonia"

    def test_run_populates_entities(self, sample_files):
        audio_path, image_path = sample_files
        pipeline = self._make_pipeline()
        result = pipeline.run(audio_path=audio_path, image_path=image_path)

        assert "symptoms" in result.named_entities
        assert "cough" in result.named_entities["symptoms"]
        assert "amoxicillin" in result.named_entities["medications"]

    def test_run_populates_vision(self, sample_files):
        audio_path, image_path = sample_files
        pipeline = self._make_pipeline()
        result = pipeline.run(audio_path=audio_path, image_path=image_path)

        assert "Pneumonia" in result.pathology_scores
        assert result.pathology_scores["Pneumonia"] == pytest.approx(0.85)
        assert "Pneumonia" in result.top_findings
        assert result.feature_vector_dim == 1024

    def test_run_populates_rag(self, sample_files):
        audio_path, image_path = sample_files
        pipeline = self._make_pipeline()
        result = pipeline.run(audio_path=audio_path, image_path=image_path)

        assert len(result.literature_references) == 1
        assert result.literature_references[0]["pmid"] == "99999"

    def test_run_includes_latency_metrics(self, sample_files):
        audio_path, image_path = sample_files
        pipeline = self._make_pipeline()
        result = pipeline.run(audio_path=audio_path, image_path=image_path)

        assert result.total_latency_s >= 0   # mocked calls complete instantly
        assert result.audio_duration_s == pytest.approx(1.0)
        assert result.input_tokens == 512
        assert result.output_tokens == 256


# ── Pydantic model tests ──────────────────────────────────────────────────────

class TestResponseModel:
    def test_default_construction(self):
        resp = ClinicalAssistantResponse(
            transcription="test",
            soap_note=SOAPNote(subjective="s", objective="o", assessment="a", plan="p"),
        )
        assert resp.confidence == "medium"
        assert resp.drug_interactions == []
        assert resp.differential_diagnoses == []

    def test_serializes_to_dict(self):
        resp = ClinicalAssistantResponse(
            transcription="test",
            soap_note=SOAPNote(subjective="s", objective="o", assessment="a", plan="p"),
            primary_diagnosis="Pneumonia",
        )
        d = resp.model_dump()
        assert d["primary_diagnosis"] == "Pneumonia"
        assert "soap_note" in d
