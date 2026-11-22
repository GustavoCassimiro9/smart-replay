import base64
import datetime
import hashlib
import os
import signal
import time
import uuid
import threading
import cv2
import pygame

from camera.camera import VideoCapture
from camera.detector import PoseDetector
from processing.bufferManager import add_frame_to_buffer, get_buffer_frames
from processing.videoSaver import save_video_with_watermark
from server.services.partidas_service import salvar_partida, salvar_video, finalizar_partida
import server.server as server

COOLDOWN_PERIOD = 5  # segundos mínimos entre dois disparos do gesto
SESSION_DURATION = 6000  # segundos até a partida ser encerrada automaticamente (~100 min)
SOUND_REPLAY = "static/replay.mp3"
SOUND_START  = "static/start.mp3"

# Câmera: índice da webcam ("0") ou URL de stream (câmera IP, DroidCam, webcam_server.py)
_source = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_SOURCE = int(_source) if _source.isdigit() else _source
CAMERA_FPS = float(os.environ["CAMERA_FPS"]) if os.environ.get("CAMERA_FPS") else None
# Sem tela (ex.: Docker): sem janela do OpenCV; o preview fica em /preview no navegador
HEADLESS = os.environ.get("HEADLESS") == "1"
PREVIEW_EVERY_N_FRAMES = 3

stop_event = threading.Event()

try:
    pygame.mixer.init()
    SOUND_ENABLED = True
except pygame.error as e:
    SOUND_ENABLED = False
    print(f"Sem dispositivo de áudio ({e}); sons desativados.")


def play_sound(path: str) -> None:
    """Toca um arquivo de som em outra thread, para nunca travar o loop principal."""
    if not SOUND_ENABLED:
        return

    def _play():
        try:
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Erro ao tocar som: {e}")
    threading.Thread(target=_play, daemon=True).start()


def frame_to_base64(frame) -> str:
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def preview_frame(frame, codigo: str, players: list[int]):
    """Cópia reduzida do frame com o código da partida (nunca desenha no frame do buffer)."""
    h, w = frame.shape[:2]
    out = cv2.resize(frame, (640, int(h * 640 / w)))
    text = f"Partida: {codigo}"
    cv2.rectangle(out, (0, 0), (640, 30), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if players:
        cv2.putText(out, "X detectado", (500, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    return out


def save_replay(partida_id, frames, fps: float, thumbnail_base64: str) -> None:
    """Codifica o clipe e registra no banco (roda em uma thread em segundo plano)."""
    save_path = save_video_with_watermark(fps, frames)
    if save_path:
        print(f"Vídeo salvo: {save_path}")
        salvar_video(partida_id, save_path, thumbnail_base64, datetime.datetime.now())
    else:
        print("Erro ao salvar o vídeo.")


def gesture_detection() -> None:
    """Loop principal da partida: captura → buffer → inferência de pose → salva o clipe no gesto."""
    cap = VideoCapture(source=CAMERA_SOURCE, fps=CAMERA_FPS)
    detector = PoseDetector(inference_width=640, skip_frames=2)

    codigo = hashlib.md5(str(uuid.uuid1()).encode()).hexdigest()
    partida = {
        "codigo": codigo,
        "pagamento": False,
        "data_inicio": datetime.datetime.now(),
        "data_fim": None,
        "videos": [],
    }
    partida_id = salvar_partida(partida)
    print(f"Partida iniciada | Codigo: {codigo}")
    play_sound(SOUND_START)

    start_time = time.time()
    last_trigger_time = 0.0
    hands_raised_since: float | None = None  # instante em que o gesto começou
    SUSTAINED_SECONDS = 2.0
    frame_count = 0
    pending_saves: list[threading.Thread] = []

    try:
        while not stop_event.is_set():
            success, frame = cap.read()
            if not success:
                print("Erro ao capturar frame da câmera.")
                continue
            frame_count += 1

            # Guarda o frame no buffer circular, para exportar o vídeo depois
            add_frame_to_buffer(frame)

            # Envia para a inferência em segundo plano (pula frames internamente)
            detector.submit(frame)

            players_raised = detector.get_result()
            current_time = time.time()

            if players_raised:
                if hands_raised_since is None:
                    hands_raised_since = current_time  # começa a contar o tempo
            else:
                hands_raised_since = None  # zera se o gesto parar

            sustained = (
                hands_raised_since is not None
                and (current_time - hands_raised_since) >= SUSTAINED_SECONDS
            )

            if sustained and (current_time - last_trigger_time) > COOLDOWN_PERIOD:
                last_trigger_time = current_time
                hands_raised_since = None  # zera para exigir mais 2 s no próximo disparo
                play_sound(SOUND_REPLAY)
                print(f"Gesto X detectado (jogadores: {players_raised})! Salvando vídeo...")

                # Copia o buffer agora; a codificação roda em segundo plano para a detecção continuar
                saver = threading.Thread(
                    target=save_replay,
                    args=(partida_id, get_buffer_frames(), cap.fps, frame_to_base64(frame)),
                    name="ReplaySaver",
                )
                saver.start()
                pending_saves.append(saver)
                pending_saves[:] = [t for t in pending_saves if t.is_alive()]

            quit_pressed = False
            if HEADLESS:
                if frame_count % PREVIEW_EVERY_N_FRAMES == 0:
                    server.publish_preview(preview_frame(frame, codigo, players_raised))
            else:
                cv2.imshow("Instaplay", frame)
                quit_pressed = cv2.waitKey(1) & 0xFF == ord('q')

            if quit_pressed or current_time - start_time > SESSION_DURATION:
                break

    finally:
        for saver in pending_saves:  # espera os clipes em gravação terminarem antes de desligar
            saver.join(timeout=60)
        finalizar_partida(partida_id, datetime.datetime.now())
        print(f"Partida finalizada | Codigo: {codigo}")
        detector.stop()
        cap.release()
        if not HEADLESS:
            cv2.destroyAllWindows()


def run_flask_server() -> None:
    server.app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)


def _request_stop(signum, frame) -> None:
    # Ctrl+C ou `docker stop`: encerra a partida direito em vez de matar o processo
    stop_event.set()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    thread_server = threading.Thread(target=run_flask_server, name="FlaskThread", daemon=True)
    thread_gesture = threading.Thread(target=gesture_detection, name="GestureThread")

    thread_server.start()
    thread_gesture.start()

    while thread_gesture.is_alive():  # espera com timeout para os sinais (Ctrl+C) serem tratados
        thread_gesture.join(timeout=0.5)
    os._exit(0)  # encerra o Flask (e outras threads) à força quando o loop da partida termina
