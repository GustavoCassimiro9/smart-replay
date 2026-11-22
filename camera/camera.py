import cv2
import threading
import queue
import time

RECONNECT_AFTER_FAILURES = 30  # leituras com falha seguidas antes de reabrir a fonte


class VideoCapture:
    """
    Captura de vídeo em thread própria — uma thread em segundo plano lê os
    frames continuamente, para o loop principal nunca ficar travado esperando
    a câmera.

    A fila interna guarda no máximo 2 frames; os antigos são descartados para
    quem consome sempre receber a imagem mais recente.

    Se a fonte parar de entregar frames (câmera desconectada, stream ainda não
    iniciado), ela é reaberta automaticamente.
    """

    def __init__(self, source=0, width: int = 1280, height: int = 720, fps: float | None = None):
        self._source = source
        self._width = width
        self._height = height
        self.cap = self._open()
        # Streams costumam informar o FPS errado; `fps` permite definir o valor real
        self.fps: float = fps or self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="VideoCaptureThread")
        self._thread.start()

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self._source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        return cap

    # ------------------------------------------------------------------
    # Loop de captura interno (roda na thread em segundo plano)
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        failures = 0
        warned = False
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                failures += 1
                if failures >= RECONNECT_AFTER_FAILURES:
                    if not warned:
                        print(f"Câmera sem sinal ({self._source}); tentando reconectar...")
                        warned = True
                    self.cap.release()
                    time.sleep(1.0)
                    self.cap = self._open()
                    failures = 0
                else:
                    time.sleep(0.03)
                continue
            if warned:
                print("Câmera conectada.")
                warned = False
            failures = 0
            # Mantém a fila atualizada — descarta o frame mais antigo se estiver cheia
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def read(self):
        """
        Retorna (True, frame) com o frame capturado mais recente.
        Espera até 1 segundo; retorna (False, None) se nada chegar.
        """
        try:
            return True, self._queue.get(timeout=1.0)
        except queue.Empty:
            return False, None

    def release(self) -> None:
        """Para a thread de captura e libera a câmera."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.cap.release()
