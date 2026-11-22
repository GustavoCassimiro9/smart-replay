import threading
import queue
import cv2
import numpy as np
from config import POSE_MODEL, POSE_MODEL_URLS, POSE_INPUT_SIZE, POSE_RUNTIME
from camera.pose_utils import has_arms_crossed_x

# ---------------------------------------------------------------------------
# Modelo único (singleton) — carregado uma vez e reaproveitado
# ---------------------------------------------------------------------------
# RTMO (OpenMMLab, Apache-2.0): estimativa de pose de várias pessoas em uma
# única etapa, executado pelo rtmlib + ONNX Runtime. Uma passada encontra
# todas as pessoas do frame, então o custo quase não muda com o número de
# jogadores em quadra.

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # verificação dupla, para carregar uma vez só entre threads
                from rtmlib import RTMO

                # Baixado no primeiro uso para ~/.cache/rtmlib
                _model = RTMO(
                    POSE_MODEL_URLS[POSE_MODEL],
                    model_input_size=POSE_INPUT_SIZE,
                    backend=POSE_RUNTIME,
                    device="cpu",
                )
    return _model


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _resize_for_inference(frame: np.ndarray, width: int = 640) -> np.ndarray:
    """Reduz o frame para `width` pixels de largura (mantendo a proporção)."""
    h, w = frame.shape[:2]
    if w == width:
        return frame
    new_h = int(h * width / w)
    return cv2.resize(frame, (width, new_h), interpolation=cv2.INTER_LINEAR)


def _detect_people(model, frame: np.ndarray) -> np.ndarray:
    """
    Roda o RTMO e retorna os pontos no formato (N_pessoas, 17, 3) — colunas
    x, y, confiança — na ordem COCO, que é o formato esperado pelo pose_utils.
    """
    keypoints, scores = model(frame)  # (N, 17, 2), (N, 17)
    people = np.concatenate([keypoints, scores[..., None]], axis=-1)
    # O rtmlib devolve uma única "pessoa" toda zerada quando ninguém é detectado
    return people[scores.max(axis=1) > 0]


# ---------------------------------------------------------------------------
# PoseDetector
# ---------------------------------------------------------------------------

class PoseDetector:
    """
    Roda a inferência de pose do RTMO em uma thread em segundo plano.

    Uso
    ---
    detector = PoseDetector()
    detector.submit(frame)          # não bloqueia — descarta o frame se estiver ocupado
    players = detector.get_result() # índices das pessoas fazendo o gesto X
    detector.stop()
    """

    def __init__(self, inference_width: int = 640, skip_frames: int = 2):
        self._model = _get_model()
        self._inference_width = inference_width
        self._skip_frames = skip_frames

        # Fila de uma posição — só interessa o frame *mais recente*
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
    # API pública
    # ------------------------------------------------------------------

    def submit(self, frame: np.ndarray) -> None:
        """
        Envia um frame para a inferência de pose.
        Pula frames conforme `skip_frames` para reduzir o uso de CPU.
        Se a thread de inferência ainda estiver ocupada, o frame é descartado.
        """
        self._frame_counter += 1
        if self._frame_counter % self._skip_frames != 0:
            return
        # Substitui qualquer frame pendente — mantém sempre o mais novo
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put(frame.copy())

    def get_result(self) -> list[int]:
        """
        Retorna a lista de índices (a partir de 0) das pessoas fazendo o gesto X.
        Não bloqueia — devolve na hora o último resultado calculado.
        """
        with self._result_lock:
            return list(self._result)

    def stop(self) -> None:
        """Sinaliza para a thread de inferência terminar e espera por ela."""
        self._stop.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Loop de inferência em segundo plano
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
