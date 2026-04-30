from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from src.audio.audio_utils import load_audio, compute_waveform_stats

# Model size guide for RTX 4060 8GB (all run on CPU, GPU free for LLM):
#   tiny.en  → 75MB,  ~1s,  good for clean audio
#   base.en  → 145MB, ~2s,  recommended default
#   small.en → 244MB, ~4s,  better accuracy, still fast on CPU
#   medium   → 769MB, ~10s, use if accuracy matters more than speed
MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"  # int8 on CPU is fastest; use float16 if running on GPU


class WhisperTranscriber:
    """
    faster-whisper wrapper. Runs on CPU — keeps GPU VRAM free for the LLM.
    Load once at startup via get_transcriber().
    """

    def __init__(self, model_size: str = MODEL_SIZE, device: str = "cpu"):
        self.model_size = model_size
        self.device = device
        self.compute_type = "int8" if device == "cpu" else "float16"
        self._model: Optional[WhisperModel] = None

    def load(self) -> None:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self.load()
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "en",
        beam_size: int = 5,
    ) -> dict:
        """
        Transcribe an audio file.

        Returns:
            {
                "text": str,
                "segments": list[dict],
                "language": str,
                "duration_s": float,
                "latency_s": float,
                "waveform_stats": dict,
            }
        """
        audio_path = Path(audio_path)
        waveform, sr = load_audio(audio_path)
        stats = compute_waveform_stats(waveform)

        t0 = time.perf_counter()
        segments_gen, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=True,          # skip silent regions
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segments = [
            {"start": s.start, "end": s.end, "text": s.text.strip()}
            for s in segments_gen
        ]
        latency = time.perf_counter() - t0
        full_text = " ".join(s["text"] for s in segments).strip()

        return {
            "text": full_text,
            "segments": segments,
            "language": info.language,
            "duration_s": round(len(waveform) / sr, 2),
            "latency_s": round(latency, 3),
            "waveform_stats": stats,
        }


_transcriber: Optional[WhisperTranscriber] = None


def get_transcriber(model_size: str = MODEL_SIZE) -> WhisperTranscriber:
    """Return the singleton transcriber, loading on first call."""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber(model_size=model_size, device="cpu")
        _transcriber.load()
    return _transcriber
