import threading
import time
from collections import deque

import cv2

from config import BUFFER_SIZE, BUFFER_JPEG_QUALITY

# Os frames ficam guardados em JPEG: um frame 720p sem compressão ocupa ~2,7 MB
# (~2,6 GB para 32 s a 30 fps), enquanto em JPEG fica em torno de 100–200 KB.
# Cada item é (horário da captura, jpg): o horário permite cortar o clipe
# exatamente antes do momento em que o gesto começou.
buffer = deque(maxlen=BUFFER_SIZE)
_lock = threading.Lock()  # a thread de gravação copia o buffer enquanto o loop adiciona frames
_jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), BUFFER_JPEG_QUALITY]


def add_frame_to_buffer(frame, timestamp: float | None = None):
    """Guarda o frame em JPEG com o horário (time.time()) em que foi capturado."""
    ok, jpg = cv2.imencode(".jpg", frame, _jpeg_params)
    if ok:
        with _lock:
            buffer.append((timestamp if timestamp is not None else time.time(), jpg))


def get_buffer_frames():
    """Cópia do buffer como lista de (horário, jpg), ainda em JPEG (barata de copiar)."""
    with _lock:
        return list(buffer)


def decode_frame(jpg):
    return cv2.imdecode(jpg, cv2.IMREAD_COLOR)
