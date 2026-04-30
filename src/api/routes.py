from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import ClinicalResponseSchema, HealthResponse
from src.fusion.multimodal_pipeline import get_pipeline

router = APIRouter(prefix="/api/v1", tags=["clinical"])

ALLOWED_AUDIO = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}
MAX_AUDIO_MB = 50
MAX_IMAGE_MB = 20


def _validate_file(upload: UploadFile, allowed: set[str], max_mb: int) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"File type '{suffix}' not supported. Allowed: {sorted(allowed)}",
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    pipeline = get_pipeline.__wrapped__ if hasattr(get_pipeline, "__wrapped__") else None
    loaded = pipeline is not None
    return HealthResponse(status="ok", models_loaded=loaded)


@router.post("/analyze", response_model=ClinicalResponseSchema)
async def analyze(
    audio: UploadFile = File(..., description="Doctor's voice recording (WAV/MP3/etc.)"),
    image: UploadFile = File(..., description="Medical image (JPEG/PNG/DICOM)"),
    language: str = Form("en"),
    imaging_modality: str = Form("Chest X-Ray"),
) -> ClinicalResponseSchema:
    """
    Full synchronous inference endpoint.
    Accepts audio + image, returns a structured clinical SOAP note as JSON.
    """
    _validate_file(audio, ALLOWED_AUDIO, MAX_AUDIO_MB)
    _validate_file(image, ALLOWED_IMAGE, MAX_IMAGE_MB)

    audio_suffix = Path(audio.filename or "audio").suffix or ".wav"
    image_suffix = Path(image.filename or "image").suffix or ".jpg"

    with (
        tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False) as af,
        tempfile.NamedTemporaryFile(suffix=image_suffix, delete=False) as imf,
    ):
        af.write(await audio.read())
        imf.write(await image.read())
        audio_path = af.name
        image_path = imf.name

    try:
        pipeline = get_pipeline()
        result = await asyncio.to_thread(
            pipeline.run,
            audio_path=audio_path,
            image_path=image_path,
            language=language,
            imaging_modality=imaging_modality,
        )
        return ClinicalResponseSchema(**result.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        Path(audio_path).unlink(missing_ok=True)
        Path(image_path).unlink(missing_ok=True)


@router.post("/analyze/stream")
async def analyze_stream(
    audio: UploadFile = File(...),
    image: UploadFile = File(...),
    language: str = Form("en"),
    imaging_modality: str = Form("Chest X-Ray"),
) -> EventSourceResponse:
    """
    SSE streaming endpoint. Streams LLM tokens as they are generated,
    then sends a final 'done' event with the full structured JSON.
    """
    _validate_file(audio, ALLOWED_AUDIO, MAX_AUDIO_MB)
    _validate_file(image, ALLOWED_IMAGE, MAX_IMAGE_MB)

    audio_suffix = Path(audio.filename or "audio").suffix or ".wav"
    image_suffix = Path(image.filename or "image").suffix or ".jpg"

    audio_bytes = await audio.read()
    image_bytes = await image.read()

    async def event_generator() -> AsyncIterator[dict]:
        af = tempfile.NamedTemporaryFile(suffix=audio_suffix, delete=False)
        imf = tempfile.NamedTemporaryFile(suffix=image_suffix, delete=False)
        try:
            af.write(audio_bytes)
            af.flush()
            imf.write(image_bytes)
            imf.flush()
            audio_path = af.name
            image_path = imf.name
            af.close()
            imf.close()

            pipeline = get_pipeline()

            # Steps 1 & 2: transcription + vision (run in thread, yield progress events)
            yield {"event": "progress", "data": json.dumps({"step": "transcribing"})}
            asr_result = await asyncio.to_thread(
                pipeline._transcriber.transcribe, audio_path, language
            )
            transcription = asr_result["text"]
            yield {"event": "transcription", "data": json.dumps({"text": transcription})}

            yield {"event": "progress", "data": json.dumps({"step": "encoding_image"})}
            vision_result = await asyncio.to_thread(pipeline._encoder.encode, image_path)
            yield {"event": "vision", "data": json.dumps({
                "top_findings": vision_result["top_findings"],
                "pathology_scores": vision_result["pathology_scores"],
            })}

            # Step 3: RAG
            yield {"event": "progress", "data": json.dumps({"step": "retrieving_literature"})}
            rag_query = f"{transcription} {' '.join(vision_result['top_findings'])}"
            rag_results = await asyncio.to_thread(pipeline._rag.retrieve, rag_query, 3)
            rag_context = pipeline._rag.format_context(rag_results)

            # Step 4: LLM streaming
            yield {"event": "progress", "data": json.dumps({"step": "generating_note"})}
            pathology_str = "\n".join(
                f"  - {p}: {s:.1%}"
                for p, s in sorted(vision_result["pathology_scores"].items(), key=lambda x: -x[1])
                if s > 0.1
            )

            streamer = pipeline._generator.stream(
                transcription=transcription,
                pathology_findings=pathology_str,
                rag_context=rag_context,
                imaging_modality=imaging_modality,
            )

            full_text = ""
            for token in streamer:
                full_text += token
                yield {"event": "token", "data": json.dumps({"token": token})}

            # Step 5: NER
            ner_result = await asyncio.to_thread(pipeline._ner.extract, full_text)
            entity_html = pipeline._ner.highlight_html(full_text, ner_result)

            yield {"event": "done", "data": json.dumps({
                "transcription": transcription,
                "raw_note": full_text,
                "top_findings": vision_result["top_findings"],
                "pathology_scores": vision_result["pathology_scores"],
                "named_entities": {k: v for k, v in ner_result.items()
                                   if k not in ("drug_interactions", "entity_spans")},
                "drug_interactions": ner_result.get("drug_interactions", []),
                "entity_html": entity_html,
                "literature_references": rag_results,
            })}

        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
        finally:
            Path(af.name).unlink(missing_ok=True)
            Path(imf.name).unlink(missing_ok=True)

    return EventSourceResponse(event_generator())
