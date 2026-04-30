from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.audio.transcriber import get_transcriber
from src.vision.image_encoder import get_image_encoder
from src.llm.note_generator import get_note_generator
from src.llm.rag_pipeline import get_rag_pipeline
from src.ner.entity_extractor import get_entity_extractor


class SOAPNote(BaseModel):
    subjective: str = ""
    objective: str = ""
    assessment: str = ""
    plan: str = ""


class ClinicalAssistantResponse(BaseModel):
    transcription: str
    soap_note: SOAPNote
    primary_diagnosis: str = ""
    differential_diagnoses: list[str] = Field(default_factory=list)
    image_findings_summary: str = ""
    recommended_medications: list[str] = Field(default_factory=list)
    follow_up: str = ""
    confidence: str = "medium"

    gradcam_overlay_b64: str = ""

    named_entities: dict = Field(default_factory=dict)
    drug_interactions: list[dict] = Field(default_factory=list)
    entity_html: str = ""

    pathology_scores: dict = Field(default_factory=dict)
    top_findings: list[str] = Field(default_factory=list)
    feature_vector_dim: int = 0

    literature_references: list[dict] = Field(default_factory=list)

    audio_duration_s: float = 0.0
    image_shape: list[int] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float = 0.0


# RTX 4060 8GB layout:
#   GPU → Ollama (Mistral 7B Q4 = ~4.1 GB VRAM)
#   CPU → Whisper, DenseNet121, sentence-transformers, spaCy


class MultimodalClinicalPipeline:
    """
    Orchestrates all modalities into one inference call.
    LLM runs on GPU via Ollama. Everything else runs on CPU.
    """

    def __init__(
        self,
        whisper_model: str = "small",
        ollama_model: str = "mistral:7b-instruct",
        rag_index_path: str = "data/pubmed_index/faiss_index",
    ):
        self.whisper_model  = whisper_model
        self.ollama_model   = ollama_model
        self.rag_index_path = rag_index_path

        self._transcriber = None
        self._encoder     = None
        self._generator   = None
        self._rag         = None
        self._ner         = None

    def load_all(self) -> None:
        """Load all components once at startup."""
        print("Loading NER (spaCy)...")
        self._ner = get_entity_extractor()

        print("Loading vision encoder (DenseNet121, CPU)...")
        self._encoder = get_image_encoder(device="cpu")

        print("Loading Whisper (faster-whisper small, CPU)...")
        self._transcriber = get_transcriber(self.whisper_model)

        print("Loading RAG pipeline (FAISS, CPU)...")
        self._rag = get_rag_pipeline(self.rag_index_path)

        print("Connecting to Ollama LLM (GPU)...")
        self._generator = get_note_generator(self.ollama_model)

        print("All components ready.")

    def run(
        self,
        audio_path: str | Path,
        image_path: str | Path,
        language: str = "en",
        imaging_modality: str = "Chest X-Ray",
    ) -> ClinicalAssistantResponse:
        if self._transcriber is None:
            self.load_all()

        wall_start = time.perf_counter()

        # Step 1: Transcribe audio (CPU)
        asr_result    = self._transcriber.transcribe(audio_path, language=language)
        transcription = asr_result["text"]
        audio_duration = asr_result["duration_s"]

        # Step 2: Encode image + GradCAM (CPU)
        vision_result   = self._encoder.encode(image_path)
        pathology_scores = vision_result["pathology_scores"]
        top_findings     = vision_result["top_findings"]
        img_meta         = vision_result["image_metadata"]

        gradcam_b64 = ""
        gradcam_img = vision_result.get("gradcam_overlay")
        if gradcam_img is not None:
            buf = io.BytesIO()
            gradcam_img.save(buf, format="PNG")
            gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        pathology_str = "\n".join(
            f"  - {p}: {s:.1%}"
            for p, s in sorted(pathology_scores.items(), key=lambda x: -x[1])
            if s > 0.1
        ) or "No significant findings above threshold."

        # Step 3: RAG retrieval (CPU)
        rag_query   = f"{transcription} {' '.join(top_findings)}"
        rag_results = self._rag.retrieve(rag_query, k=3)
        rag_context = self._rag.format_context(rag_results)

        # Step 4: Generate SOAP note (GPU via Ollama)
        llm_result = self._generator.generate(
            transcription=transcription,
            pathology_findings=pathology_str,
            rag_context=rag_context,
            imaging_modality=imaging_modality,
        )
        parsed    = llm_result["parsed"]
        soap_data = parsed.get("soap_note", {})

        # Step 5: NER + drug interaction check (CPU)
        ner_result  = self._ner.extract(llm_result["raw_text"])
        entity_html = self._ner.highlight_html(llm_result["raw_text"], ner_result)

        total_latency = time.perf_counter() - wall_start

        return ClinicalAssistantResponse(
            transcription=transcription,
            soap_note=SOAPNote(**{k: soap_data.get(k, "") for k in ["subjective", "objective", "assessment", "plan"]}),
            primary_diagnosis=parsed.get("primary_diagnosis", ""),
            differential_diagnoses=parsed.get("differential_diagnoses", []),
            image_findings_summary=parsed.get("image_findings_summary", ""),
            recommended_medications=parsed.get("recommended_medications", []),
            follow_up=parsed.get("follow_up", ""),
            confidence=parsed.get("confidence", "medium"),
            gradcam_overlay_b64=gradcam_b64,
            named_entities={k: v for k, v in ner_result.items()
                            if k not in ("drug_interactions", "entity_spans", "_clean_text")},
            drug_interactions=ner_result.get("drug_interactions", []),
            entity_html=entity_html,
            pathology_scores=pathology_scores,
            top_findings=top_findings,
            feature_vector_dim=len(vision_result["feature_vector"]),
            literature_references=rag_results,
            audio_duration_s=round(audio_duration, 2),
            image_shape=[img_meta["height"], img_meta["width"]],
            input_tokens=llm_result["input_tokens"],
            output_tokens=llm_result["output_tokens"],
            total_latency_s=round(total_latency, 3),
        )


_pipeline: Optional[MultimodalClinicalPipeline] = None


def get_pipeline(**kwargs) -> MultimodalClinicalPipeline:
    """Return the singleton pipeline, loading all models on first call."""
    global _pipeline
    if _pipeline is None:
        _pipeline = MultimodalClinicalPipeline(**kwargs)
        _pipeline.load_all()
    return _pipeline
