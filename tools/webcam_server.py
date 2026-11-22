"""
Roda no COMPUTADOR (Windows): transmite a webcam como stream MJPEG para o
container Docker (que não acessa dispositivos USB) conseguir ler por HTTP.

Os frames são enviados numa taxa fixa (repetindo frames se a câmera for mais
lenta), então o tempo do vídeo sempre bate com o tempo real — gravações feitas
a partir deste stream mantêm a velocidade certa para a regra dos 2 segundos.

Uso:    python webcam_server.py [--camera 0] [--port 8090] [--fps 30]
Stream: http://localhost:8090/video
"""
import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Compartilha a webcam como stream MJPEG.")
    p.add_argument("--camera", type=int, default=0, help="índice da câmera (padrão: 0)")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--fps", type=float, default=30.0, help="taxa fixa do stream (padrão: 30)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--quality", type=int, default=85, help="qualidade JPEG (padrão: 85)")
    return p.parse_args()


class Camera:
    """Lê a webcam em uma thread em segundo plano e guarda o frame mais recente."""

    def __init__(self, index, width, height):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise SystemExit(f"Não foi possível abrir a câmera {index}. Tente --camera 1.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def latest(self):
        with self.lock:
            return self.frame


def make_handler(camera, fps, quality):
    interval = 1.0 / fps
    encode = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # não polui o terminal com cada requisição
            pass

        def do_GET(self):
            if self.path == "/":
                body = b'<html><body style="margin:0;background:#000"><img src="/video" style="width:100%"></body></html>'
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path != "/video":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            print(f"Cliente conectado: {self.client_address[0]}")
            next_t = time.perf_counter()
            try:
                while True:
                    frame = camera.latest()
                    if frame is not None:
                        ok, jpg = cv2.imencode(".jpg", frame, encode)
                        if ok:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                            self.wfile.write(jpg.tobytes())
                            self.wfile.write(b"\r\n")
                    next_t += interval
                    time.sleep(max(0.0, next_t - time.perf_counter()))
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                print(f"Cliente desconectado: {self.client_address[0]}")

    return Handler


def main():
    args = parse_args()
    camera = Camera(args.camera, args.width, args.height)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(camera, args.fps, args.quality))
    print(f"Webcam {args.camera} em http://localhost:{args.port}/video ({args.fps:g} fps). Ctrl+C para parar.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
