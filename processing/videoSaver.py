import datetime
import shutil
import subprocess
import uuid

import cv2
import numpy as np

from config import VIDEO_PATH, WATERMARK_PATH, GESTURE_TRIM_SECONDS
from processing.bufferManager import decode_frame

# H.264 pelo ffmpeg abre em navegadores e celulares; o "mp4v" do próprio OpenCV
# (usado quando o ffmpeg não está instalado) gera arquivos maiores e não abre no navegador.
FFMPEG = shutil.which("ffmpeg")


def generate_new_path_name():
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    unique_key = str(uuid.uuid1())
    return f"{now}-{unique_key}.mp4"


class _FfmpegWriter:
    """Envia os frames BGR brutos para o ffmpeg, que codifica em H.264 (yuv420p, faststart para tocar na web)."""

    def __init__(self, path, fps, width, height):
        self.proc = subprocess.Popen(
            [
                FFMPEG, "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # o H.264 exige largura e altura pares
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame):
        self.proc.stdin.write(frame.tobytes())

    def release(self) -> bool:
        self.proc.stdin.close()
        return self.proc.wait() == 0


class _OpenCvWriter:
    def __init__(self, path, fps, width, height):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    def write(self, frame):
        self.out.write(frame)

    def release(self) -> bool:
        ok = self.out.isOpened()
        self.out.release()
        return ok


def _load_watermark(frame_width, frame_height):
    """Logo redimensionada para 15% da largura do frame, com as máscaras de mistura já calculadas."""
    logo = cv2.imread(WATERMARK_PATH, cv2.IMREAD_UNCHANGED)  # RGBA
    if logo is None:
        return None

    target_width = max(80, int(frame_width * 0.15))
    logo_h, logo_w = logo.shape[:2]
    target_height = int(logo_h * target_width / logo_w)
    interp = cv2.INTER_AREA if target_width < logo_w else cv2.INTER_CUBIC
    logo = cv2.resize(logo, (target_width, target_height), interpolation=interp)

    # Máscaras em float32 evitam artefatos de arredondamento
    b, g, r, alpha = cv2.split(logo)
    overlay = cv2.merge((b, g, r)).astype(np.float32)
    mask = alpha.astype(np.float32) / 255.0
    mask = np.stack([mask, mask, mask], axis=2)  # (H, W, 3)

    logo_h, logo_w = logo.shape[:2]
    x = frame_width - logo_w - 20
    y = frame_height - logo_h - 20
    return overlay, mask, 1.0 - mask, x, y, logo_w, logo_h


def save_video_with_watermark(fps, frames):
    """
    Exporta um clipe de replay com a marca d'água.

    `frames` é uma cópia do buffer, em JPEG, vinda de bufferManager.get_buffer_frames().
    Os últimos GESTURE_TRIM_SECONDS são descartados para o clipe não terminar no gesto.
    Retorna o nome do arquivo dentro de VIDEO_PATH, ou None se der erro.
    """
    if not frames:
        print("Buffer vazio. Nenhum vídeo para salvar.")
        return None

    trim = int(fps * GESTURE_TRIM_SECONDS)
    if len(frames) > trim:
        frames = frames[:-trim]

    first = decode_frame(frames[0])
    frame_height, frame_width = first.shape[:2]

    watermark = _load_watermark(frame_width, frame_height)
    if watermark is None:
        print("Erro ao carregar a marca d'água. Verifique o caminho do arquivo.")
        return None
    overlay, mask, mask_inv, x, y, logo_w, logo_h = watermark

    file_name = generate_new_path_name()
    writer_cls = _FfmpegWriter if FFMPEG else _OpenCvWriter
    writer = writer_cls(f"{VIDEO_PATH}{file_name}", fps, frame_width, frame_height)

    try:
        for jpg in frames:
            frame = decode_frame(jpg)
            if frame is None or frame.shape[:2] != (frame_height, frame_width):
                continue  # pula frames corrompidos ou uma mudança de resolução no meio do buffer
            roi = frame[y:y + logo_h, x:x + logo_w].astype(np.float32)
            frame[y:y + logo_h, x:x + logo_w] = (roi * mask_inv + overlay * mask).astype(np.uint8)
            writer.write(frame)
    finally:
        ok = writer.release()

    if not ok:
        print("Erro ao gravar o vídeo.")
        return None

    codec = "H.264" if FFMPEG else "mp4v (instale o ffmpeg para H.264)"
    print(f"Vídeo salvo ({len(frames) / fps:.0f}s, {codec}).")
    return file_name
