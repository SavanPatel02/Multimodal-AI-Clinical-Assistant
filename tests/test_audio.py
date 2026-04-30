"""Tests for audio transcription pipeline."""
from __future__ import annotations

import io
import os
import struct
import tempfile
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.audio.audio_utils import (
    compute_waveform_stats,
    get_audio_duration,
    load_audio,
    save_waveform_bytes,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_wav(duration_s: float = 1.0, sr: int = 16000) -> str:
    """Write a synthetic sine-wave WAV to a temp file, return path."""
    t = np.linspace(0, duration_s, int(sr * duration_s), dtype=np.float32)
    samples = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((samples * 32767).astype(np.int16).tobytes())
    return tmp.name


@pytest.fixture
def wav_path(tmp_path):
    path = make_wav(duration_s=2.0)
    yield path
    Path(path).unlink(missing_ok=True)


# ── audio_utils tests ─────────────────────────────────────────────────────────

class TestLoadAudio:
    def test_loads_wav_returns_float32(self, wav_path):
        waveform, sr = load_audio(wav_path)
        assert waveform.dtype == np.float32
        assert sr == 16000
        assert len(waveform) > 0

    def test_duration_approx(self, wav_path):
        waveform, sr = load_audio(wav_path)
        assert abs(len(waveform) / sr - 2.0) < 0.1

    def test_unsupported_format_raises(self, tmp_path):
        bad = tmp_path / "audio.xyz"
        bad.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported"):
            load_audio(bad)


class TestWaveformStats:
    def test_stats_keys(self, wav_path):
        waveform, _ = load_audio(wav_path)
        stats = compute_waveform_stats(waveform)
        assert "rms" in stats
        assert "peak" in stats
        assert "duration_samples" in stats
        assert "is_clipped" in stats

    def test_rms_positive(self, wav_path):
        waveform, _ = load_audio(wav_path)
        stats = compute_waveform_stats(waveform)
        assert stats["rms"] > 0

    def test_silent_audio(self):
        silence = np.zeros(16000, dtype=np.float32)
        stats = compute_waveform_stats(silence)
        assert stats["rms"] == 0.0
        assert not stats["is_clipped"]


class TestAudioDuration:
    def test_duration(self, wav_path):
        dur = get_audio_duration(wav_path)
        assert abs(dur - 2.0) < 0.2


class TestSaveWaveformBytes:
    def test_returns_bytes(self, wav_path):
        waveform, sr = load_audio(wav_path)
        data = save_waveform_bytes(waveform, sr)
        assert isinstance(data, bytes)
        assert len(data) > 44  # WAV header is 44 bytes


# ── Transcriber tests (mocked) ─────────────────────────────────────────────────

class TestWhisperTranscriber:
    def test_transcribe_returns_text(self, wav_path):
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 1.0
        mock_segment.text = " Patient reports chest pain."

        mock_info = MagicMock()
        mock_info.language = "en"

        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)

        with patch("faster_whisper.WhisperModel", return_value=mock_model):
            from src.audio.transcriber import WhisperTranscriber
            t = WhisperTranscriber(model_size="small", device="cpu")
            t._model = mock_model
            result = t.transcribe(wav_path)

        assert "text" in result
        assert result["text"] == "Patient reports chest pain."
        assert "latency_s" in result
        assert "waveform_stats" in result
        assert "duration_s" in result
