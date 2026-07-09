#!/usr/bin/env python3
"""Trim + normalize a clip for a slightly cleaner LipNet *upload* demo (still no accuracy guarantee).

Keeps **AAC audio** when the input has an audio stream (otherwise video-only MP4).

Run from the project root, then upload the output MP4 in the Streamlit "Upload your video" tab.

Example:
  python tools/prep_demo_video.py ~/Movies/talk.mp4 ./demo_upload.mp4
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="Any video readable by ffmpeg")
    p.add_argument("output", type=Path, help="Where to write MP4 (e.g. demo_upload.mp4)")
    p.add_argument(
        "--seconds",
        type=float,
        default=4.0,
        help="Length from the start of the file (default: 4)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=720,
        help="Scale width; height keeps aspect (default: 720)",
    )
    p.add_argument(
        "--no-audio",
        action="store_true",
        help="Strip audio (smaller file; browser preview will be silent).",
    )
    args = p.parse_args()

    if not args.input.is_file():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 1

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("ffmpeg not found in PATH. Install it (e.g. brew install ffmpeg).", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={args.width}:-2:flags=lanczos,fps=25,setsar=1"
    if args.no_audio:
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(args.input),
            "-t",
            str(args.seconds),
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    else:
        # Optional audio: omit mapping when source has no audio (ffmpeg 4+).
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(args.input),
            "-t",
            str(args.seconds),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "+faststart",
            str(args.output),
        ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Wrote {args.output.resolve()}")
    if args.no_audio:
        print("Tip: output has no audio (--no-audio). Re-run without that flag to keep sound from the source.")
    else:
        print("Tip: audio is kept when the source has an audio stream; film face-on, ~4s; transcript may still be wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
