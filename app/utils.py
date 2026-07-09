import os
import shutil
import urllib.request

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import tensorflow as tf
from typing import List
import cv2
import numpy as np
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

GRID_W, GRID_H = 360, 288
MOUTH_SLICE = (slice(190, 236), slice(80, 220))


def _resize_to_grid_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Stretch each frame to GRID_W×GRID_H (matches original LipNet / GRID training)."""
    return cv2.resize(frame_bgr, (GRID_W, GRID_H), interpolation=cv2.INTER_AREA)


def _letterbox_to_grid_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Fit ``frame_bgr`` into GRID_W×GRID_H without stretching (black bars).

    Used for **upload** preprocessing so portrait phone video is not squashed before
    MediaPipe. GRID ``.mpg`` clips use ``_resize_to_grid_bgr`` instead to match training.
    """
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    scale = min(GRID_W / w, GRID_H / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    x0 = (GRID_W - nw) // 2
    y0 = (GRID_H - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _read_grid_mouth_frame_tensors(path: str) -> List[tf.Tensor]:
    """Decode GRID-style video: stretch to grid, grayscale, fixed mouth crop per frame.

    Uses ``CAP_PROP_FRAME_COUNT`` when OpenCV reports it (matches ``load_video`` /
    training-style iteration). If that yields no frames, reads until EOF.
    """
    cap = cv2.VideoCapture(path)
    frames: List[tf.Tensor] = []
    n_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_prop > 0:
        for _ in range(n_prop):
            ret, frame = cap.read()
            if not ret:
                break
            frame = _resize_to_grid_bgr(frame)
            g = tf.image.rgb_to_grayscale(tf.convert_to_tensor(frame))
            frames.append(g[MOUTH_SLICE[0], MOUTH_SLICE[1], :])
    if not frames:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _resize_to_grid_bgr(frame)
            g = tf.image.rgb_to_grayscale(tf.convert_to_tensor(frame))
            frames.append(g[MOUTH_SLICE[0], MOUTH_SLICE[1], :])
    cap.release()
    return frames


def load_video(path: str) -> tf.Tensor:
    frames = _read_grid_mouth_frame_tensors(path)
    if not frames:
        raise ValueError(f"No frames could be read from: {path}")
    stacked = tf.stack(frames, axis=0)
    mean = tf.math.reduce_mean(stacked)
    std = tf.math.reduce_std(tf.cast(stacked, tf.float32))
    return tf.cast((stacked - mean), tf.float32) / std


def _resample_frame_list(frames: List[tf.Tensor], target: int) -> List[tf.Tensor]:
    """Bring a variable-length mouth sequence to ``target`` frames for LipNet.

    GRID clips often decode to slightly more than 75 frames in OpenCV. Training used
    fixed-length batches of 75; **linear index resampling** smears motion and hurts
    CTC. We **center-crop** when longer than ``target``, and **pad** with the last
    frame when shorter (same length, contiguous real frames when possible).
    """
    n = len(frames)
    if n == 0:
        raise ValueError("No frames could be read from the video file.")
    if n == target:
        return frames
    if n > target:
        start = (n - target) // 2
        return frames[start : start + target]
    out = list(frames)
    last = frames[-1]
    while len(out) < target:
        out.append(last)
    return out


def load_inference_video(path: str, target_frames: int = 75) -> tf.Tensor:
    """Load and normalize a GRID ``.mpg`` clip for LipNet (stretch → mouth crop → 75 frames).

    Frame iteration matches ``load_video`` (``CAP_PROP_FRAME_COUNT`` when available) so
    the same pixels are seen as in training helpers, then resampled to ``target_frames``.
    """
    frames = _read_grid_mouth_frame_tensors(path)
    if not frames:
        raise ValueError(f"No frames could be read from: {path}")
    frames = _resample_frame_list(frames, target_frames)
    stacked = tf.stack(frames, axis=0)
    mean = tf.math.reduce_mean(stacked)
    std = tf.math.reduce_std(tf.cast(stacked, tf.float32))
    return tf.cast((stacked - mean), tf.float32) / std


# MediaPipe Face Mesh indices covering outer / inner lip contour (canonical model).
_MOUTH_MESH_IDX = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415,
)


def _grid_mouth_from_letterboxed(boxed_bgr: np.ndarray) -> np.ndarray:
    """GRID mouth window on an image already letterboxed to GRID size."""
    gray = cv2.cvtColor(boxed_bgr, cv2.COLOR_BGR2GRAY)
    patch = gray[MOUTH_SLICE[0], MOUTH_SLICE[1]].astype(np.float32)
    return np.expand_dims(patch, axis=-1)


def _grid_mouth_fallback(frame_bgr: np.ndarray) -> np.ndarray:
    """Fixed GRID crop after letterboxing (matches training layout better than squash-resize)."""
    return _grid_mouth_from_letterboxed(_letterbox_to_grid_bgr(frame_bgr))


def _mouth_patch_from_task_landmarks(
    frame_bgr: np.ndarray, face_landmarks: list, pad_ratio: float = 0.2
) -> np.ndarray | None:
    """Crop mouth from MediaPipe Tasks face landmarks (478-point topology)."""
    h, w = frame_bgr.shape[:2]
    xs: List[int] = []
    ys: List[int] = []
    for idx in _MOUTH_MESH_IDX:
        lm = face_landmarks[idx]
        if lm.x is None or lm.y is None:
            return None
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))
    span_x = max(xs) - min(xs) + 1
    span_y = max(ys) - min(ys) + 1
    pad_x = max(4, int(pad_ratio * span_x))
    pad_y = max(4, int(pad_ratio * span_y))
    x1 = max(0, min(xs) - pad_x)
    x2 = min(w, max(xs) + pad_x)
    y1 = max(0, min(ys) - pad_y)
    y2 = min(h, max(ys) + pad_y)
    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None
    patch = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (140, 46), interpolation=cv2.INTER_AREA)
    return np.expand_dims(resized.astype(np.float32), axis=-1)


_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_face_landmarker_model() -> Path:
    """Download MediaPipe Face Landmarker task file once (newer wheels omit ``solutions``)."""
    path = Path(__file__).resolve().parent / "face_landmarker.task"
    if path.exists() and path.stat().st_size > 512_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".task.download")
    try:
        with urllib.request.urlopen(_FACE_LANDMARKER_URL, timeout=120) as resp:
            with open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
        tmp.replace(path)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe face landmarker model (~3.5 MB). "
            "Check your network, or save the file manually as "
            f"{path} (see MediaPipe face landmarker model zoo)."
        ) from e
    return path


def _resample_np_frames(frames: List[np.ndarray], target: int) -> List[np.ndarray]:
    """Temporal trim/pad to ``target`` frames (center crop if longer; pad last if shorter)."""
    n = len(frames)
    if n == 0:
        raise ValueError("No frames could be read from the video file.")
    if n == target:
        return frames
    if n > target:
        start = (n - target) // 2
        return frames[start : start + target]
    out = list(frames)
    last = frames[-1]
    while len(out) < target:
        out.append(last.copy())
    return out


def load_inference_upload_auto(path: str, target_frames: int = 75) -> tf.Tensor:
    """Build model input from an arbitrary upload using face-tracked mouth crops.

    Each frame is **letterboxed** to 360×288 (same canvas as GRID) before Face Landmarker
    runs, so portrait phone video is not horizontally squashed.

    Uses MediaPipe **Tasks** Face Landmarker (video mode) — compatible with current
    ``mediapipe`` wheels that no longer expose ``mediapipe.solutions``.

    On first use, downloads ``face_landmarker.task`` next to this module (~3.5 MB).

    The model was trained on GRID-style crops; auto-reframing improves usability but
    does not guarantee accurate transcripts on all poses or lighting.
    """
    from mediapipe.tasks.python.core import base_options as mp_base
    from mediapipe.tasks.python.vision import face_landmarker as mp_face
    from mediapipe.tasks.python.vision.core import image as mp_image
    from mediapipe.tasks.python.vision.core import vision_task_running_mode as mp_run

    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps < 1.0:
        fps = 25.0
    bgr_frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        bgr_frames.append(frame)
    cap.release()
    if not bgr_frames:
        raise ValueError("No frames could be read from the video file.")

    model_path = str(_ensure_face_landmarker_model())
    base = mp_base.BaseOptions(
        model_asset_path=model_path,
        delegate=mp_base.BaseOptions.Delegate.CPU,
    )
    opts = mp_face.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_run.VisionTaskRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_face.FaceLandmarker.create_from_options(opts)
    crops: List[np.ndarray] = []
    try:
        last_good: np.ndarray | None = None
        for i, frame in enumerate(bgr_frames):
            boxed = _letterbox_to_grid_bgr(frame)
            rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
            mp_img = mp_image.Image(mp_image.ImageFormat.SRGB, rgb)
            ts_ms = int(i * 1000.0 / fps)
            result = landmarker.detect_for_video(mp_img, ts_ms)
            patch: np.ndarray | None = None
            if result.face_landmarks:
                patch = _mouth_patch_from_task_landmarks(boxed, result.face_landmarks[0])
            if patch is None:
                patch = (
                    last_good
                    if last_good is not None
                    else _grid_mouth_from_letterboxed(boxed)
                )
            else:
                last_good = patch
            crops.append(patch)
    finally:
        landmarker.close()

    crops = _resample_np_frames(crops, target_frames)
    stacked = np.stack(crops, axis=0)
    t = tf.convert_to_tensor(stacked, dtype=tf.float32)
    mean = tf.math.reduce_mean(t)
    std = tf.math.reduce_std(t)
    return (t - mean) / std

def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    normalized = path.replace("\\", "/")
    file_name = normalized.split("/")[-1].split(".")[0]
    video_path = str(_PROJECT_ROOT / "data" / "s1" / f"{file_name}.mpg")
    alignment_path = str(_PROJECT_ROOT / "data" / "alignments" / "s1" / f"{file_name}.align")
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments