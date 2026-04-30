"""Tests for LLM note generation and RAG pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.llm.prompts import SOAP_GENERATION_TEMPLATE, SYSTEM_PROMPT


# ── Prompt template tests ─────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_soap_template_formatting(self):
        rendered = SOAP_GENERATION_TEMPLATE.format(
            transcription="Patient has chest pain.",
            imaging_modality="Chest X-Ray",
            pathology_findings="  - Pneumonia: 85%",
            rag_context="[1] Paper about pneumonia",
        )
        assert "chest pain" in rendered
        assert "Chest X-Ray" in rendered
        assert "Pneumonia" in rendered

    def test_system_prompt_not_empty(self):
        assert len(SYSTEM_PROMPT) > 50


# ── NoteGenerator tests (mocked) ─────────────────────────────────────────────

class TestClinicalNoteGenerator:
    MOCK_OUTPUT = json.dumps({
        "soap_note": {
            "subjective": "Patient reports chest pain.",
            "objective": "X-ray shows consolidation.",
            "assessment": "Community-acquired pneumonia.",
            "plan": "Start amoxicillin 500mg TID.",
        },
        "primary_diagnosis": "Community-Acquired Pneumonia",
        "differential_diagnoses": ["Pulmonary Edema", "Pleuritis"],
        "image_findings_summary": "Right lower lobe consolidation.",
        "recommended_medications": ["amoxicillin", "ibuprofen"],
        "follow_up": "Repeat CXR in 6 weeks.",
        "confidence": "high",
        "literature_references": [],
    })

    def _make_generator(self):
        from src.llm.note_generator import ClinicalNoteGenerator

        mock_response = MagicMock()
        mock_response.message.content = self.MOCK_OUTPUT
        mock_response.prompt_eval_count = 200
        mock_response.eval_count = 150

        mock_client = MagicMock()
        mock_client.list.return_value = MagicMock(models=[MagicMock(model="mistral:7b-instruct")])
        mock_client.chat.return_value = mock_response

        gen = ClinicalNoteGenerator(model="mistral:7b-instruct")
        gen._client = mock_client
        gen._verified = True
        return gen

    def test_generate_returns_parsed_json(self):
        gen = self._make_generator()
        result = gen.generate(
            transcription="Patient has fever.",
            pathology_findings="Consolidation: 80%",
            rag_context="PubMed abstract 1",
        )
        assert "raw_text" in result
        assert "parsed" in result
        assert "input_tokens" in result
        assert "output_tokens" in result
        assert not result["parsed"].get("parse_error", False)

    def test_extract_json_valid(self):
        from src.llm.note_generator import ClinicalNoteGenerator
        result = ClinicalNoteGenerator._extract_json(self.MOCK_OUTPUT)
        assert result["primary_diagnosis"] == "Community-Acquired Pneumonia"

    def test_extract_json_invalid_returns_raw(self):
        from src.llm.note_generator import ClinicalNoteGenerator
        result = ClinicalNoteGenerator._extract_json("no json here at all")
        assert result.get("parse_error") is True


# ── RAG pipeline tests (mocked) ──────────────────────────────────────────────

class TestPubMedRAGPipeline:
    def test_build_and_retrieve(self, tmp_path):
        with (
            patch("src.llm.rag_pipeline.HuggingFaceEmbeddings") as MockEmb,
            patch("src.llm.rag_pipeline.FAISS") as MockFAISS,
        ):
            mock_emb = MagicMock()
            MockEmb.return_value = mock_emb

            mock_vs = MagicMock()
            mock_doc = MagicMock()
            mock_doc.page_content = "Pneumonia treatment abstract."
            mock_doc.metadata = {"pmid": "12345", "title": "Pneumonia Study"}
            mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.85)]
            mock_vs.save_local = MagicMock()
            MockFAISS.from_documents.return_value = mock_vs

            from src.llm.rag_pipeline import PubMedRAGPipeline
            rag = PubMedRAGPipeline(index_path=tmp_path / "idx", top_k=1)
            rag._embeddings = mock_emb

            abstracts = [
                {"pmid": "12345", "title": "Pneumonia Study", "abstract": "This study shows..."},
            ]
            rag.build_index(abstracts)
            rag._vectorstore = mock_vs

            results = rag.retrieve("chest infection pneumonia")

        assert len(results) == 1
        assert results[0]["pmid"] == "12345"
        assert results[0]["score"] == pytest.approx(0.85, abs=0.01)

    def test_format_context(self):
        from src.llm.rag_pipeline import PubMedRAGPipeline
        rag = PubMedRAGPipeline()
        results = [
            {"title": "Study A", "pmid": "111", "text": "Abstract A.", "score": 0.9},
            {"title": "Study B", "pmid": "222", "text": "Abstract B.", "score": 0.7},
        ]
        ctx = rag.format_context(results)
        assert "Study A" in ctx
        assert "111" in ctx
        assert "Abstract B." in ctx
