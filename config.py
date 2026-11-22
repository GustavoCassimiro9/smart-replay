# config.py
FPS = 30  # Frames por segundo
CLIP_SECONDS = 15  # Duração do replay salvo
GESTURE_HOLD_SECONDS = 2  # Tempo segurando o X até disparar o replay
GESTURE_MARGIN_SECONDS = 1.0  # O clipe termina este tempo antes do X ser reconhecido (movimento de levantar os braços)
GESTURE_RELEASE_SECONDS = 0.7  # O X só conta como desfeito se sumir por mais que isso (o modelo "pisca")
GESTURE_COOLDOWN_SECONDS = 5  # Intervalo mínimo entre dois replays
# Buffer: clipe + tempo segurando o X + margem + intervalo entre replays (o disparo pode esperar por ele)
BUFFER_SIZE = int(FPS * (CLIP_SECONDS + GESTURE_HOLD_SECONDS + GESTURE_MARGIN_SECONDS + GESTURE_COOLDOWN_SECONDS))  # 23 s
BUFFER_JPEG_QUALITY = 90  # Frames ficam em JPEG no buffer (~20x menos memória que sem compressão)
VIDEO_RESOLUTION = (1280, 720)  # Resolução do vídeo
VIDEO_PATH = "static/videos/"
WATERMARK_PATH = "static/logo.png"

# Modelo de pose — RTMO (OpenMMLab, Apache-2.0) via rtmlib + ONNX Runtime.
# "rtmo-s" é o mais leve; "rtmo-m" e "rtmo-l" são mais precisos e mais lentos.
POSE_MODEL = "rtmo-s"
POSE_MODEL_URLS = {
    "rtmo-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip",
    "rtmo-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
    "rtmo-l": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip",
}
POSE_INPUT_SIZE = (640, 640)
POSE_RUNTIME = "onnxruntime"  # "openvino" pode ser mais rápido em CPUs Intel (pip install openvino)
