"""One shared RapidOCR engine for every reader, tuned to stay light while the game runs.

The defaults keep a read from competing with the game for the CPU:
- One ONNX Runtime thread. The library's default uses every core: on a 12-core / 24-thread CPU a read
  finished ~3x sooner but kept ~16 cores busy (2 s of CPU per read instead of 0.4 s).
- The detector only ever shrinks its input (limit_type "max", 960 px). RapidOCR's default ("min") enlarges
  small crops before detecting, which costs more than it gains on game text.
- No text-direction classifier: game text is never upside down, and the classifier flips some lines.
On an NVIDIA GPU (device="cuda") a small crop reads in ~35 ms with about one core busy.

Images are numpy arrays in BGR order, as mss and OpenCV give them.
"""

import logging
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class EngineSettings:
    device: str = "cpu"           # "cpu" or "cuda" (NVIDIA GPU)
    threads: int = 1              # ONNX Runtime threads on the CPU; 0 = the library's default (all cores)
    det_limit_side: int = 960     # the detector's input size limit, in pixels
    det_limit_type: str = "max"   # "max": shrink so the long side fits; "min": grow so the short side reaches it

    def __post_init__(self):
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")
        if self.threads < 0:
            raise ValueError(f"threads must be >= 0, got {self.threads}")
        if self.det_limit_type not in ("max", "min"):
            raise ValueError(f"det_limit_type must be 'max' or 'min', got {self.det_limit_type!r}")

    def params(self) -> dict:
        """The settings as RapidOCR's `params`."""
        params = {
            "Global.log_level": "error",  # an empty read logs a warning otherwise; CUDA fallback is checked below
            "Global.use_cls": False,
            "Det.limit_type": self.det_limit_type,
            "Det.limit_side_len": self.det_limit_side,
        }
        if self.threads:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = self.threads
            params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
        if self.device == "cuda":
            params["EngineConfig.onnxruntime.use_cuda"] = True
            # The default (EXHAUSTIVE) re-tunes the kernels for every new text-line width.
            params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "HEURISTIC"
        return params


@dataclass(frozen=True)
class Row:
    """One line of text, in pixels of the image that was passed in (any upscaling undone)."""
    text: str
    score: float   # recognition confidence, 0..1
    x: float       # centre
    y: float       # centre
    height: float
    left: float
    right: float


def upscale(image: np.ndarray, min_height: int) -> tuple[np.ndarray, float]:
    """`image` grown (Lanczos, same aspect) so it is at least `min_height` px tall, and the factor used.
    Small game text reads more reliably when it's bigger."""
    h, w = image.shape[:2]
    if not min_height or h >= min_height or h == 0:
        return image, 1.0
    factor = min_height / h
    size = (max(1, round(w * factor)), min_height)
    return np.asarray(Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)), factor


def rows_from(boxes, texts, scores, factor: float = 1.0) -> list[Row]:
    """RapidOCR's boxes (4 corner points each) and texts as Rows in reading order (lines top to bottom, left to
    right within a line), scaled back by `factor`."""
    if boxes is None or texts is None:
        return []
    rows = []
    for box, text, score in zip(np.asarray(boxes, dtype=float).reshape(-1, 4, 2), texts, scores or ()):
        text = str(text).strip()
        if not text:
            continue
        xs, ys = box[:, 0] / factor, box[:, 1] / factor
        rows.append(Row(text, float(score), float(xs.mean()), float(ys.mean()), float(ys.max() - ys.min()),
                        float(xs.min()), float(xs.max())))
    # Boxes on one line differ a little in height and centre, so group by line before ordering by x.
    lines: list[list[Row]] = []
    for row in sorted(rows, key=lambda r: r.y):
        if lines and abs(row.y - lines[-1][0].y) <= 0.5 * max(row.height, lines[-1][0].height):
            lines[-1].append(row)
        else:
            lines.append([row])
    return [row for line in lines for row in sorted(line, key=lambda r: r.x)]


class OcrEngine:
    """Loads the models on first use and serializes reads (RapidOCR isn't safe to call from two threads)."""

    def __init__(self, settings: EngineSettings = EngineSettings()):
        self.settings = settings
        self._ocr = None
        self._failed = False
        self._checked_providers = False
        self._lock = threading.Lock()

    def read(self, image: np.ndarray, *, min_height: int = 0) -> list[Row] | None:
        """The lines of text in `image` (BGR), top to bottom. Images shorter than `min_height` px are enlarged
        first; rows come back in the original image's pixels. None when OCR can't load (logged once)."""
        with self._lock:
            ocr = self._load()
            if ocr is None:
                return None
            big, factor = upscale(np.ascontiguousarray(image), min_height)
            out = ocr(big)
            if not self._checked_providers:
                self._check_providers(ocr)
            return rows_from(out.boxes, out.txts, out.scores, factor)

    def providers(self) -> dict[str, str]:
        """Which execution provider each loaded model runs on, e.g. {"det": "CUDA", "rec": "CUDA"}."""
        with self._lock:
            return _session_providers(self._ocr) if self._ocr is not None else {}

    def _load(self):
        if self._ocr is None and not self._failed:
            try:
                if self.settings.device == "cuda":
                    _add_conda_dll_dir()
                from rapidocr import RapidOCR  # imported late: the package stays importable without it
                self._ocr = RapidOCR(params=self.settings.params())
            except Exception as e:
                self._failed = True
                log.warning("OCR unavailable: %s", e)
        return self._ocr

    def _check_providers(self, ocr) -> None:
        # The models load on the first read, and RapidOCR falls back to the CPU with only a log line.
        self._checked_providers = True
        found = _session_providers(ocr)
        if self.settings.device == "cuda" and any(p != "CUDA" for p in found.values()):
            log.warning("CUDA was asked for but OCR runs on %s (driver or CUDA libraries missing?)", found)


_shared: dict[EngineSettings, OcrEngine] = {}
_shared_lock = threading.Lock()


def shared(settings: EngineSettings = EngineSettings()) -> OcrEngine:
    """The one engine for these settings, so every reader shares the loaded models."""
    with _shared_lock:
        if settings not in _shared:
            _shared[settings] = OcrEngine(settings)
        return _shared[settings]


def _session_providers(ocr) -> dict[str, str]:
    found = {}
    for name, attr in (("det", "text_det"), ("rec", "text_rec")):
        session = getattr(getattr(getattr(ocr, attr, None), "session", None), "session", None)
        if session is not None:
            found[name] = session.get_providers()[0].removesuffix("ExecutionProvider")
    return found


def _add_conda_dll_dir() -> None:
    # conda keeps the env's CUDA/cuDNN DLLs in Library\bin and puts it on PATH only when the env is activated;
    # onnxruntime finds its CUDA provider's DLLs through PATH. Running the env's python.exe directly needs this.
    conda_bin = Path(sys.prefix, "Library", "bin")
    path = os.environ.get("PATH", "")
    if sys.platform == "win32" and conda_bin.is_dir() and str(conda_bin).lower() not in path.lower():
        os.environ["PATH"] = f"{conda_bin}{os.pathsep}{path}"
