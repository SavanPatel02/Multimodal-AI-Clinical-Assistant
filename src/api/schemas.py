from __future__ import annotations

from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    language: str = "en"
    imaging_modality: str = "Chest X-Ray"


class SOAPNoteSchema(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str


class DrugInteractionWarning(BaseModel):
    drug_a: str
    drug_b: str
    severity: str  # major | moderate | minor
    description: str
    recommendation: str


class LiteratureReference(BaseModel):
    title: str
    pmid: str
    text: str
    score: float


class ClinicalResponseSchema(BaseModel):
    transcription: str
    soap_note: SOAPNoteSchema
    primary_diagnosis: str = ""
    differential_diagnoses: list[str] = Field(default_factory=list)
    image_findings_summary: str = ""
    recommended_medications: list[str] = Field(default_factory=list)
    follow_up: str = ""
    confidence: str = "medium"

    gradcam_overlay_b64: str = ""   # base64-encoded PNG of GradCAM heatmap

    named_entities: dict = Field(default_factory=dict)
    drug_interactions: list[DrugInteractionWarning] = Field(default_factory=list)
    entity_html: str = ""

    pathology_scores: dict = Field(default_factory=dict)
    top_findings: list[str] = Field(default_factory=list)

    literature_references: list[LiteratureReference] = Field(default_factory=list)

    audio_duration_s: float = 0.0
    image_shape: list[int] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_latency_s: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"
