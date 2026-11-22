import threading
import queue
import cv2
import numpy as np
from config import POSE_MODEL, POSE_MODEL_URLS, POSE_INPUT_SIZE, POSE_RUNTIME
from camera.pose_utils import has_arms_crossed_x

# ---------------------------------------------------------------------------
# Singleton model — loaded once, reused forever
# ---------------------------------------------------------------------------
# RTMO (OpenMMLab, Apache-2.0): one-stage multi-person pose estimation, run
# through rtmlib + ONNX Runtime. One pass finds every person in the frame, so
# the cost barely changes with the number of players on court.

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # double-checked locking
                from rtmlib import RTMO

                # Downloaded on first use to ~/.cache/rtmlib
                _model = RTMO(
                    POSE_MODEL_URLS[POSE_MODEL],
                    model_input_size=POSE_INPUT_SIZE,
                    backend=POSE_RUNTIME,
                    device="cpu",
                )
    return _model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resize_for_inference(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Downscale frame to `width` pixels wide (aspect ratio preserved)."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(h * width / w)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def _detect_people(model, frame: np.ndarray) -> np.ndarray:
    """
    Run RTMO and return keypoints as (N_persons, 17, 3) — columns x, y,
    confidence — in COCO order, the format pose_utils expects.
    """
    keypoints, scores = model(frame)  # (N, 17, 2), (N, 17)
    people = np.concatenate([keypoints, scores[..., None]], axis=-1)
    # rtmlib returns a single all-zero "person" when nobody is detected
    return people[scores.max(axis=1) > 0]


# ---------------------------------------------------------------------------
# PoseDetector
# ---------------------------------------------------------------------------

class PoseDetector:
    """
    Runs RTMO pose inference in a background thread.

    Usage
    -----
    detector = PoseDetector()
    detector.submit(frame)          # non-blocking — drop frame if busy
    players = detector.get_result() # list of person indices making the X gesture
    detector.stop()
    """

    def __init__(self, inference_width: int = 640, skip_frames: int = 2):
        self._model = _get_model()
        self._inference_width = inference_width
        self._skip_frames = skip_frames

        # Single-slot queue — we only care about the *latest* frame
        self._frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self._result: list[int] = []
        self._result_lock = threading.Lock()

        self._frame_counter = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="PoseDetectorThread"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, frame: np.ndarray) -> None:
        """
        Send a frame for pose inference.
        Frames are skipped according to `skip_frames` to reduce CPU load.
        If the inference thread is still busy the frame is silently dropped.
        """
        self._frame_counter += 1
        if self._frame_counter % self._skip_frames != 0:
            return
        # Replace any pending frame — always keep the newest
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put(frame.copy())

    def get_result(self) -> list[int]:
        """
        Return the list of person indices (0-based) making the X gesture.
        Non-blocking — returns the last computed result immediately.
        """
        with self._result_lock:
            return list(self._result)

    def stop(self) -> None:
        """Signal the inference thread to exit and wait for it."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Background inference loop
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            small = _resize_for_inference(frame, self._inference_width)
            people = _detect_people(self._model, small)

            raised = [idx for idx, kps in enumerate(people) if has_arms_crossed_x(kps)]

            with self._result_lock:
                self._result = raised
