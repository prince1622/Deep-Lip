import os
import re
import sys
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path


def _configure_hf_hub_httpx() -> None:
    """Hub uses its own ``httpx.Client``; point ``verify=`` at certifi (or off if insecure)."""
    try:
        import httpx
        from huggingface_hub.utils import close_session, set_async_client_factory, set_client_factory
        from huggingface_hub.utils._http import (
            async_hf_request_event_hook,
            async_hf_response_event_hook,
            hf_request_event_hook,
        )
    except ImportError:
        return

    insecure = os.environ.get("LIPREADER_INSECURE_SSL", "").strip() in ("1", "true", "yes")
    if insecure:
        verify: bool | str = False
    else:
        try:
            import certifi

            verify = certifi.where()
        except ImportError:
            verify = True

    def _sync_factory():
        return httpx.Client(
            event_hooks={"request": [hf_request_event_hook]},
            follow_redirects=True,
            timeout=None,
            verify=verify,
        )

    def _async_factory():
        return httpx.AsyncClient(
            event_hooks={
                "request": [async_hf_request_event_hook],
                "response": [async_hf_response_event_hook],
            },
            follow_redirects=True,
            timeout=None,
            verify=verify,
        )

    close_session()
    set_client_factory(_sync_factory)
    set_async_client_factory(_async_factory)


def _apply_ssl_for_hub_downloads() -> None:
    """TLS for stdlib + Hugging Face Hub (Whisper weight downloads)."""
    insecure = os.environ.get("LIPREADER_INSECURE_SSL", "").strip() in ("1", "true", "yes")
    if insecure:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context  # noqa: S501
    else:
        try:
            import truststore

            truststore.inject_into_ssl()
        except (ImportError, AttributeError, RuntimeError, OSError):
            try:
                import certifi
                import ssl

                ca = certifi.where()
                os.environ["SSL_CERT_FILE"] = ca
                os.environ["REQUESTS_CA_BUNDLE"] = ca
                os.environ["CURL_CA_BUNDLE"] = ca

                def _https_ctx_certifi() -> ssl.SSLContext:
                    return ssl.create_default_context(cafile=ca)

                ssl._create_default_https_context = _https_ctx_certifi  # type: ignore[assignment]
            except ImportError:
                pass

        try:
            import certifi

            ca = certifi.where()
            os.environ["SSL_CERT_FILE"] = ca
            os.environ["REQUESTS_CA_BUNDLE"] = ca
            os.environ["CURL_CA_BUNDLE"] = ca
        except ImportError:
            pass

    _configure_hf_hub_httpx()


_apply_ssl_for_hub_downloads()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import imageio
import numpy as np
import streamlit as st
import tensorflow as tf

from modelutil import load_model
from transcribe_audio import transcribe_with_faster_whisper
from utils import load_inference_upload_auto, load_inference_video, num_to_char

st.set_page_config(
    page_title="Deep Lip",
    page_icon=":material/transcribe:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_S1 = _PROJECT_ROOT / "data" / "s1"
_APP_DIR = Path(__file__).resolve().parent
_PREVIEW_GIF = _APP_DIR / "preview_mouth.gif"
_PREVIEW_DISPLAY_MP4 = _APP_DIR / "preview_display.mp4"
_SESSION_UPLOAD = _APP_DIR / "_session_upload.mp4"
_UPLOAD_WIDGET_KEY = "lipreader_video_uploader"

# Typical GRID session filenames in this repo (e.g. bbaf2n.mpg): four letters, digit, a/n/p/s, .mpg
_GRID_CORPUS_NAME = re.compile(r"^[a-z]{4}[0-9][anps]\.mpg$", re.IGNORECASE)


def _dataset_clip_is_typical_grid_corpus(name: str) -> bool:
    return bool(_GRID_CORPUS_NAME.fullmatch(name.strip()))


def _ffmpeg_executable() -> str:
    override = os.environ.get("FFMPEG_PATH")
    if override:
        return override.strip().strip('"')
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError(
        "ffmpeg not found. Install it (e.g. `brew install ffmpeg`) or set FFMPEG_PATH to the ffmpeg binary."
    )


def _source_file_has_audio(path: Path) -> bool:
    """Best-effort: True if ffprobe sees an audio stream (else False)."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe or not path.is_file():
        return False
    try:
        r = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return bool(r.stdout.strip())
    except OSError:
        return False


# Bumped when Whisper load logic changes (forces Streamlit to reload the cached model).
_WHISPER_RESOURCE_TAG = "base-en-hf-httpx-certifi-2026-05-06"


@st.cache_resource(show_spinner="Loading Whisper (first run may download English model from Hugging Face)…")
def _whisper_load_result(_tag: str = _WHISPER_RESOURCE_TAG):
    """Return ``(WhisperModel | None, error_markdown | None)``."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        return None, (
            "**faster-whisper** is not installed in this Python environment.\n\n"
            f"Run exactly (same interpreter Streamlit uses):\n\n`{sys.executable} -m pip install -U faster-whisper certifi`\n\n"
            "Then **fully stop and restart** Streamlit (not just refresh the browser)."
        )

    override = os.environ.get("LIPREADER_WHISPER_MODEL", "").strip()
    if override:
        model_ref = str(Path(override).expanduser().resolve())
        local_only = True
    else:
        model_ref = "base.en"
        local_only = False

    last_errors: list[str] = []
    for device, ctype in (
        ("auto", "int8"),
        ("auto", "float16"),
        ("cpu", "int8"),
        ("cpu", "float32"),
    ):
        try:
            kwargs = dict(device=device, compute_type=ctype)
            if local_only:
                kwargs["local_files_only"] = True
            model = WhisperModel(model_ref, **kwargs)
            return model, None
        except Exception as exc:  # noqa: BLE001
            last_errors.append(f"{device}/{ctype}: {type(exc).__name__}: {exc}")

    ssl_hint = ""
    joined = " ".join(last_errors)
    if "CERTIFICATE_VERIFY_FAILED" in joined or "SSL" in joined:
        ssl_hint = (
            "\n\n**SSL / certificate (common on macOS):**\n"
            "- The app pins **Hugging Face Hub**’s HTTP client to **certifi** (`verify=certifi.where()`). "
            "If errors persist: `python -m pip install -U certifi truststore` then restart Streamlit.\n"
            "- **truststore** still hooks the macOS keychain for other TLS; Hub downloads use certifi explicitly.\n"
            "- If you use **python.org** Python on Mac, run **Install Certificates.command** in the Python folder under Applications.\n"
            "- Or set a **local model** (no Hub download): download the CTranslate2 `base.en` files on another network, then set env  \n"
            "  `LIPREADER_WHISPER_MODEL=/path/to/model-folder` and restart.\n"
            "- **Last resort (insecure):** `export LIPREADER_INSECURE_SSL=1` then restart Streamlit — only on trusted networks.\n"
        )

    return None, (
        f"**Whisper** could not load `{model_ref}`. Details:\n\n"
        + "\n".join(f"- {e}" for e in last_errors)
        + ssl_hint
        + f"\n\nTry: `{sys.executable} -m pip install -U faster-whisper ctranslate2 certifi` then restart Streamlit."
    )


@st.cache_data(max_entries=24, show_spinner=False)
def _transcribe_upload_cached(file_digest: str, media_path: str) -> tuple[str | None, str | None]:
    """Cache by upload digest; path is stable ``_SESSION_UPLOAD`` but content changes with digest."""
    model, err = _whisper_load_result()
    if model is None:
        return None, err
    return transcribe_with_faster_whisper(Path(media_path), model=model)


def _convert_to_h264_mp4(src: Path, dest: Path) -> None:
    ffmpeg = _ffmpeg_executable()
    with_audio = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
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
        str(dest),
    ]
    try:
        subprocess.run(with_audio, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-an",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
            capture_output=True,
            text=True,
        )


@st.cache_resource
def _cached_model():
    return load_model()


_UPLOAD_PIPELINE_VER = "grid-letterbox-v2-temporal-crop"


@st.cache_data(max_entries=12, show_spinner=False)
def _upload_mouth_tensor_cached(pipeline_ver: str, file_digest: str, source_path: str) -> np.ndarray:
    """Cache auto-reframed input; ``pipeline_ver`` bumps invalidate old preprocessing."""
    _ = pipeline_ver
    t = load_inference_upload_auto(source_path)
    return t.numpy()


def _mouth_gif_from_tensor(video: tf.Tensor) -> None:
    """GIF preview: tensor is mean/std normalized — stretch per frame for display."""
    frames = [np.squeeze(f.numpy(), axis=-1) for f in video]
    vis: list[np.ndarray] = []
    for f in frames:
        lo, hi = float(np.min(f)), float(np.max(f))
        if hi - lo < 1e-6:
            vis.append(np.zeros_like(f, dtype=np.uint8))
        else:
            vis.append(np.uint8(np.clip((f - lo) / (hi - lo) * 255.0, 0, 255)))
    imageio.mimsave(_PREVIEW_GIF, vis, fps=10)


def _predict_ui(
    model,
    video: tf.Tensor,
    show_debug: bool,
    *,
    clip_context: str,
    dataset_basename: str | None = None,
    audio_transcript: str | None = None,
    audio_transcript_note: str | None = None,
    upload_lip_only_reason: str | None = None,
) -> None:
    with st.spinner("Running LipNet inference…"):
        yhat = model.predict(tf.expand_dims(video, axis=0), verbose=0)

    decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
    converted = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode("utf-8")

    if clip_context == "upload" and audio_transcript:
        st.success("Transcript ready")
        st.markdown(f"### Transcript (from your video’s audio)\n## \"{audio_transcript}\"")
        st.caption(
            "**Whisper** (`base.en`, English only) on the soundtrack — speech recognition for lip reading."
        )
    elif clip_context == "upload":
        st.success("Prediction complete")
        st.metric("Output time steps", int(yhat.shape[1]))
        st.markdown(f"### Predicted transcript (lip motion only)\n# \"{converted}\"")
        if upload_lip_only_reason == "no_audio":
            st.info(
                "**Whisper skipped** — no usable audio in this file after normalization.\n\n"
                "The line above is **LipNet only** (mouth motion → letters). It was trained on **GRID** "
                "studio clips, so on **silent or phone** video it **usually will not** match what you mouthed.\n\n"
                "**To match spoken words:** add a **mic / soundtrack** to the MP4 (e.g. AAC), turn on "
                "**Transcribe speech from audio** in the sidebar, then run **Generate transcript** again."
            )
        else:
            if audio_transcript_note:
                st.info(audio_transcript_note)
            if upload_lip_only_reason == "whisper_off":
                st.warning(
                    "This line is **lip-motion guess text** (English letters only, **no** audio used by LipNet). "
                    "Turn on **Transcribe speech from audio** in the sidebar if your file has a soundtrack — "
                    "Whisper usually matches spoken words **much** better than lip-only on MP4 uploads."
                )
            elif upload_lip_only_reason is None:
                st.warning(
                    "This line is **lip-motion guess text** (English letters only, **no** audio used by LipNet). "
                    "If Whisper did not return useful text, try clearer audio or turn **Transcribe speech from audio** "
                    "on when the clip has a soundtrack."
                )
    else:
        st.success("Prediction complete")
        st.metric("Output time steps", int(yhat.shape[1]))
        st.markdown(f"### Predicted transcript\n# \"{converted}\"")

    if clip_context == "dataset" and dataset_basename and not _dataset_clip_is_typical_grid_corpus(
        dataset_basename
    ):
        st.caption(
            "This file is **not** a typical GRID studio clip — the line above is often **unrelated** "
            "to what you said. Open **Why** below for detail."
        )
        with st.expander("Why this transcript often does not match your words", expanded=False):
            st.markdown(
                """
This clip is **not** treated as a canonical GRID studio take (for example your own **`.mp4`**
or a phone export placed under `data/s1/`).

The model maps mouth motion to **GRID-style word fragments**, not arbitrary spoken English.
If you said something like **“best efforts”** but the line looks random, that is **expected**:
lip reading here **does not use audio** and this checkpoint is **not** a general speech recognizer.

**Fair demo of readable text:** pick **`bbaf2n.mpg`** (or another `????#?.mpg` name from the corpus).

**Your own footage:** use **Upload · MP4** for **MediaPipe** mouth tracking (better framing than this tab for phone video).
                """.strip()
            )

    if show_debug:
        with st.expander("Debug: raw model outputs"):
            st.write("Argmax tokens:", tf.argmax(yhat, axis=2).numpy())
            st.write("CTC decoded token ids:", decoder)


def _render_video_preview(path: Path) -> bool:
    try:
        _convert_to_h264_mp4(path, _PREVIEW_DISPLAY_MP4)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        st.error(
            "Could not convert video with ffmpeg. Install ffmpeg (`brew install ffmpeg`) "
            "or set FFMPEG_PATH."
        )
        st.exception(e)
        return False
    st.video(_PREVIEW_DISPLAY_MP4.read_bytes())
    cap = (
        "**Browser preview:** ffmpeg H.264 (+ AAC when present). "
        "**LipNet input:** same source file decoded with **OpenCV** (pixels can differ slightly from the preview)."
    )
    if not _source_file_has_audio(path):
        cap += (
            " **This file has no audio stream** — the player will be silent. "
            "If you used `tools/prep_demo_video.py` with an older copy that used `-an`, re-run prep "
            "(audio is kept by default now) and replace `demo_upload.mp4`."
        )
    st.caption(cap)
    return True


def _presentation_expander() -> None:
    with st.expander("Presenter cheat sheet (~2 min)", expanded=False):
        st.markdown(
            """
**What to say in one breath**  
This demo shows **two ways to get text from video**: (1) **visual lip motion** with a GRID-trained LipNet-style model, and (2) on uploads with sound, **Whisper** on the audio track so the headline transcript matches what people actually said.

**Demo A — “the model behaves as trained” (slides)**  
1. Tab **Dataset · GRID** — default clip **`bbaf2n.mpg`** when present.  
2. Play preview → **Generate transcript**.  
3. Point to the **mouth GIF** (75 frames) and the **CTC transcript** — emphasize *in-distribution* GRID studio data; **LipNet path does not use audio**.

**Demo B — “real phone MP4” (honest + relatable)**  
1. Tab **Upload · MP4** — short frontal clip, good light.  
2. Optional: trim/scale with `tools/prep_demo_video.py` (keeps AAC by default; add `--no-audio` only for a smaller silent file).  
3. With **Whisper** on in the sidebar, click **Generate transcript** after watching — the **audio transcript** is the headline; the messy lip-only line stays hidden when Whisper succeeds.

**Closing line**  
Lip-only models need curated data; **Whisper** is why everyday MP4s still get a credible string in this demo.
            """.strip()
        )


def _limitations_expander() -> None:
    with st.expander("Scope: two modalities, limits, fair comparisons", expanded=False):
        st.markdown(
            """
**1 · LipNet path (always runs)**  
Reads **mouth appearance** over **75 grayscale frames** (fixed alphabet: lowercase English letters, digits, space, `'` `?` `!`).  
It is **not** multilingual output in the usual sense — and it is **not** using the soundtrack for that line.

**2 · Whisper path (uploads with audio)**  
On **Upload · MP4**, if the normalized file still has an **audio track**, the app runs **Whisper `base.en`** and shows that string as the **primary transcript** (speech recognition, English).  
**LipNet** remains a **secondary “visual-only”** line for comparison.

**3 · Why phone lip-text looks random**  
The LipNet checkpoint matches **GRID** studio mouths. Casual video is **out-of-distribution** for lip-only decoding; mismatches vs spoken words are **expected**, not a language bug.

**4 · What “better lip-only English” would take**  
New data + retraining (a full ML project), not a UI toggle.

**5 · What to show your audience**  
- **LipNet credibility:** **`bbaf2n.mpg`** on **Dataset · GRID**.  
- **Everyday MP4 → readable text:** **Upload · MP4** with **Whisper** enabled in the sidebar.
            """.strip()
        )


def _interactive_hero() -> None:
    st.markdown("### Live demo flow")
    a, b, c = st.columns(3)
    with a:
        st.info(
            "**1 · Pick a story** — **Dataset** = model on training-style video **or** **Upload** = real MP4 video."
        )
    with b:
        st.info(
            "**2 · Watch once** — preview is browser-friendly; inference still reads the **source pixels** "
            "the pipeline expects."
        )
    with c:
        st.info(
            "**3 · Reveal results** — **Dataset · GRID:** press **Generate transcript** after watching. "
            "**Upload · MP4:** watch preview, then **Generate transcript**."
        )
    st.caption(
        "**Presenter note:** **Dataset · GRID** keeps a deliberate “reveal” so you can narrate the clip before scores appear."
    )


def _sync_clip_state(prefix: str, clip_id: str) -> None:
    """Reset reveal flag when the active clip changes."""
    clip_key = f"{prefix}_clip_id"
    if st.session_state.get(clip_key) != clip_id:
        st.session_state[clip_key] = clip_id
        st.session_state[f"{prefix}_show_result"] = False


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    div[data-testid="stMetric"] { background: rgba(120,120,120,0.08); padding: 12px 16px; border-radius: 8px; }
    /* st.video: data-testid is on the <video> (or <iframe> for YouTube), not a wrapper div */
    video[data-testid="stVideo"],
    iframe[data-testid="stVideo"] {
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        max-height: min(65vh, 640px) !important;
        object-fit: contain !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        display: block !important;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        max-width: 100% !important;
        max-height: min(50vh, 520px) !important;
        height: auto !important;
        object-fit: contain !important;
        margin-left: 0 !important;
        margin-right: auto !important;
        display: block !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Deep Lip - Lip Reading using Speech recognition")

_interactive_hero()
_limitations_expander()
_presentation_expander()

with st.sidebar:
    st.header("Options")
    show_debug = st.toggle("Show debug tensors", value=False)
    use_whisper = st.toggle(
        "Transcribe speech from audio (Whisper)",
        value=True,
        help="Uses the MP4 soundtrack on **Upload · MP4**. English `base.en` model; first run downloads ~140 MB.",
    )
    st.caption(
        "First-time Whisper download needs working TLS (`certifi` / `truststore`). "
        f"Interpreter: `{sys.executable}`"
    )

_DATASET_VIDEO_SUFFIXES = frozenset({".mpg", ".mpeg", ".mp4"})

try:
    dataset_options = sorted(
        f.name
        for f in _DATA_S1.iterdir()
        if f.is_file() and f.suffix.lower() in _DATASET_VIDEO_SUFFIXES
    )
except FileNotFoundError:
    dataset_options = []

tab_ds, tab_up = st.tabs(["Dataset · GRID", "Upload · MP4"])

model = _cached_model()
if model is None:
    st.error("Failed to load the model. Check that `models/checkpoint` exists.")
    st.stop()

with tab_ds:
    if not dataset_options:
        st.warning("No video files (`.mpg`, `.mpeg`, `.mp4`) found under `data/s1`.")
    else:
        _demo_clip = "bbaf2n.mpg"
        _default_ix = (
            dataset_options.index(_demo_clip) if _demo_clip in dataset_options else 0
        )
        selected = st.selectbox(
            "GRID clip (in-distribution when the filename matches corpus style)",
            dataset_options,
            index=_default_ix,
            key="ds_select",
        )
        if selected:
            file_path = _DATA_S1 / selected
            _sync_clip_state("ds", selected)

            col_a, col_b = st.columns([2, 3], gap="large")
            with col_a:
                st.subheader("Clip preview")
                preview_ok = _render_video_preview(file_path)
                if st.button(
                    "Generate transcript",
                    key="ds_run_btn",
                    disabled=not preview_ok,
                    type="primary",
                ):
                    st.session_state.ds_show_result = True

            with col_b:
                st.subheader("Results panel")
                if not st.session_state.get("ds_show_result"):
                    st.markdown(
                        """
**After you click Generate transcript**, this column shows:

- **Mouth crop animation** — 75-frame tensor the LipNet stack sees  
- **Metrics** — sequence length after CTC decoding  
- **Predicted text** — visual lip-reading only (no audio on this tab)

_Watch the preview on the left, then press **Generate transcript**._
                        """.strip()
                    )
                else:
                    video_tensor = None
                    with st.status("Building 75-frame mouth tensor & running LipNet…", expanded=True) as status:
                        try:
                            video_tensor = load_inference_video(str(file_path))
                        except ValueError as err:
                            status.update(label="Could not read video", state="error")
                            st.error(str(err))
                        else:
                            _mouth_gif_from_tensor(video_tensor)
                            st.markdown("**Mouth region the model sees (75 frames)**")
                            st.image(str(_PREVIEW_GIF), use_container_width=True)
                            status.update(label="Frames ready", state="complete")
                    if video_tensor is not None:
                        st.divider()
                        _predict_ui(
                            model,
                            video_tensor,
                            show_debug,
                            clip_context="dataset",
                            dataset_basename=selected,
                        )

with tab_up:
    st.info(
        "**Upload flow:** watch the preview, then **Generate transcript**.\n\n"
        "**Vision-only (no audio):** LipNet maps **mouth motion → letters**. This checkpoint was trained on **GRID** "
        "studio faces — **it cannot reliably recover your exact sentence** on silent phone or casual MP4; odd text is **expected**.\n\n"
        "**For accurate text:** include a **mic / soundtrack** in the MP4 and turn on **Transcribe speech from audio** "
        "(Whisper reads sound, not lips). For a **lip-reading demo** without audio, use **Dataset · GRID** with a corpus clip."
    )
    with st.expander("How to try other videos", expanded=False):
        st.markdown(
            """
1. **Choose a file** — the app normalizes it for preview only until you press **Generate transcript**.

2. **Switch clips** — use **Try another video** under the preview to clear the uploader, then choose a different file.

3. **Dataset · GRID** — put clips under `data/s1/` as **`.mpg`**, **`.mpeg`**, or **`.mp4`**, refresh the page, then pick the name from the dropdown. (The list only scans that folder — not your whole disk. **Upload · MP4** is for files anywhere.)

4. **Slightly better odds** — a few seconds, **face large**, **looking at the camera**, bright even light, little head movement, **no heavy filters**.
            """.strip()
        )

    up = st.file_uploader(
        "Choose a short video (MP4, MOV, MPG — frontal face, good light)",
        type=["mp4", "mov", "mpg", "mpeg", "avi"],
        key=_UPLOAD_WIDGET_KEY,
    )
    if up is not None:
        raw = up.getbuffer()
        file_digest = hashlib.sha256(memoryview(raw)).hexdigest()

        suffix = Path(up.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_src = Path(tmp.name)
        try:
            had_audio_in = _source_file_has_audio(tmp_src)
            _convert_to_h264_mp4(tmp_src, _SESSION_UPLOAD)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            st.error("Could not normalize your upload for playback.")
            st.exception(e)
            tmp_src.unlink(missing_ok=True)
        else:
            if had_audio_in and not _source_file_has_audio(_SESSION_UPLOAD):
                st.warning(
                    "The normalized preview file has **no audio**, but the upload appeared to have an audio track. "
                    "ffmpeg may have failed to copy AAC (try another export / `brew reinstall ffmpeg`). "
                    "**Whisper cannot run** without a sound track on `_session_upload.mp4`."
                )
            tmp_src.unlink(missing_ok=True)

            _sync_clip_state("up", file_digest)

            col_a, col_b = st.columns([1, 1], gap="large")
            with col_a:
                st.subheader("Your upload")
                preview_ok = _render_video_preview(_SESSION_UPLOAD)
                if st.button(
                    "Generate transcript",
                    key="up_run_btn",
                    disabled=not preview_ok,
                    type="primary",
                    help="Runs Whisper (if enabled + audio) and LipNet after you have watched the preview.",
                ):
                    st.session_state.up_show_result = True

            with col_b:
                st.subheader("Results")
                st.caption(
                    "**After Generate:** LipNet always runs; **Whisper** runs when enabled and audio exists. "
                    "The **mouth-crop preview** (75 frames) always appears here, same idea as **Dataset · GRID**."
                )
                if not st.session_state.get("up_show_result"):
                    st.markdown(
                        """
**This panel will show:**

- **Mouth crop animation** — 75-frame tensor LipNet sees (MediaPipe mouth track)  
- **Transcript from audio** — Whisper `base.en` when the clip has sound and the sidebar toggle is on  
- **Headline text** — Whisper string when it succeeds; otherwise the lip-only line with guidance

_LipNet’s rough lip-only string is hidden when Whisper returns a transcript._

_Watch the preview on the left, then press **Generate transcript**._
                        """.strip()
                    )
                else:
                    video_tensor = None
                    audio_txt: str | None = None
                    audio_note: str | None = None
                    with st.status("Mouth tracking, LipNet & optional Whisper…", expanded=True) as status:
                        try:
                            arr = _upload_mouth_tensor_cached(
                                _UPLOAD_PIPELINE_VER, file_digest, str(_SESSION_UPLOAD)
                            )
                            video_tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
                        except (ImportError, RuntimeError) as err:
                            status.update(label="Face model unavailable", state="error")
                            st.error(str(err))
                        except ValueError as err:
                            status.update(label="Could not read video", state="error")
                            st.error(str(err))
                        else:
                            if use_whisper and _source_file_has_audio(_SESSION_UPLOAD):
                                with st.spinner("Transcribing speech with Whisper…"):
                                    w_text, w_err = _transcribe_upload_cached(
                                        file_digest, str(_SESSION_UPLOAD.resolve())
                                    )
                                wt = (w_text or "").strip()
                                if wt:
                                    audio_txt = wt
                                elif w_err:
                                    audio_note = w_err
                                else:
                                    audio_note = (
                                        "Whisper returned no text (quiet clip, wrong language, or unclear audio)."
                                    )
                            elif use_whisper:
                                # Explanation for silent uploads lives in _predict_ui (single info box).
                                audio_note = None
                            else:
                                audio_note = (
                                    "**Transcribe speech from audio** is off in the sidebar — "
                                    "only the lip-reading line is shown."
                                )

                            _mouth_gif_from_tensor(video_tensor)
                            st.markdown("**Mouth region sent to the model (75 frames)**")
                            st.image(str(_PREVIEW_GIF), use_container_width=True)
                            status.update(label="Frames ready", state="complete")
                    if video_tensor is not None:
                        st.divider()
                        has_audio = _source_file_has_audio(_SESSION_UPLOAD)
                        lip_reason: str | None = None
                        if not audio_txt:
                            if use_whisper and not has_audio:
                                lip_reason = "no_audio"
                            elif not use_whisper:
                                lip_reason = "whisper_off"
                        _predict_ui(
                            model,
                            video_tensor,
                            show_debug,
                            clip_context="upload",
                            audio_transcript=audio_txt,
                            audio_transcript_note=audio_note,
                            upload_lip_only_reason=lip_reason,
                        )
            st.divider()
            if st.button(
                "Try another video",
                key="reset_uploader_btn",
                help="Clears the file picker so you can choose a different clip.",
            ):
                st.session_state.pop(_UPLOAD_WIDGET_KEY, None)
                st.session_state.pop("up_show_result", None)
                st.rerun()
