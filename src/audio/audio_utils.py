import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from pydub import AudioSegment


SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
TARGET_SR = 16000


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio file and return (waveform, sample_rate)."""
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    if path.suffix.lower() == ".wav":
        waveform, sr = sf.read(str(path), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
    else:
        audio = AudioSegment.from_file(str(path))
        audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        waveform = samples / (2**15)
        sr = TARGET_SR

    if sr != TARGET_SR:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    return waveform, sr


def convert_to_wav(input_path: str | Path, output_path: str | Path | None = None) -> str:
    """Convert any supported audio format to 16kHz mono WAV."""
    input_path = Path(input_path)
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    audio = AudioSegment.from_file(str(input_path))
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    audio.export(str(output_path), format="wav")
    return str(output_path)


def get_audio_duration(path: str | Path) -> float:
    """Return duration in seconds."""
    waveform, sr = load_audio(path)
    return len(waveform) / sr


def compute_waveform_stats(waveform: np.ndarray) -> dict:
    """Return basic waveform statistics for logging."""
    return {
        "duration_samples": len(waveform),
        "rms": float(np.sqrt(np.mean(waveform**2))),
        "peak": float(np.max(np.abs(waveform))),
        "is_clipped": bool(np.any(np.abs(waveform) >= 0.99)),
    }


def save_waveform_bytes(waveform: np.ndarray, sr: int = TARGET_SR) -> bytes:
    """Serialize waveform to WAV bytes (for Streamlit audio widget)."""
    import io
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    return buf.getvalue()
