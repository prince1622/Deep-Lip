"""Optional faster-whisper transcription for video files with an audio track."""

from __future__ import annotations

from pathlib import Path


def transcribe_with_faster_whisper(media_path: Path, *, model) -> tuple[str | None, str | None]:
    """Run an already-loaded ``WhisperModel``. Returns ``(text, error)``."""
    try:
        segments, _info = model.transcribe(
            str(media_path),
            language="en",
            task="transcribe",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt="Clear spoken English.",
        )
        parts = [s.text.strip() for s in segments]
        text = " ".join(parts).strip()
        return (text or None), None
    except Exception as exc:  # noqa: BLE001 — surface any runtime failure to UI
        return None, str(exc)
