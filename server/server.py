import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, send_file
from flask_cors import CORS

from server.services.partidas_service import pegar_jogadas_por_partida, pegar_partida

app = Flask(__name__)
CORS(app)

BASE_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIDEO_DIRECTORY = os.path.join(BASE_DIRECTORY, 'static', 'videos')

# Último frame da câmera para o preview no navegador (usado quando roda sem tela)
_preview_jpg: bytes | None = None
_preview_lock = threading.Lock()


def publish_preview(frame) -> None:
    global _preview_jpg
    ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if ok:
        with _preview_lock:
            _preview_jpg = jpg.tobytes()


@app.route('/preview', methods=['GET'])
def preview():
    """Preview ao vivo da câmera (MJPEG) com o código da partida."""
    def stream():
        while True:
            with _preview_lock:
                jpg = _preview_jpg
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            time.sleep(0.1)
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/partida/video/<filename>', methods=['GET'])
def get_video(filename):
    video_path = os.path.join(VIDEO_DIRECTORY, filename)
    print(f"video_path {video_path}")
    if os.path.exists(video_path):
        print("tem video")
        response = send_file(video_path, as_attachment=True, mimetype='video/mp4')
        return response
    else:
        return jsonify({"error": "Video not found"}), 404

@app.route('/partida/<codigo>', methods=['GET'])
def obter_partida(codigo):
    print(codigo)
    try:
        partida = pegar_partida(codigo)
        jogadas = pegar_jogadas_por_partida(partida.id)
        if partida:
            return jsonify({
                'id': partida.id,
                'codigo': partida.codigo,
                'pagamento': partida.pagamento,
                'data_inicio': partida.data_inicio,
                'data_fim': partida.data_fim,
                "jogadas": jogadas
            }), 200
        else:
            return jsonify({'message': 'Partida não encontrada'}), 404
    except ValueError as e:
        return jsonify({'message': str(e)}), 400

@app.route('/partida/<partida_id>/jogadas', methods=['GET'])
def obter_jogadas_por_partida(partida_id):
    try:
        jogadas = pegar_jogadas_por_partida(partida_id)
        if jogadas:
            return jsonify(jogadas), 200
        else:
            return jsonify({'message': 'Jogadas para a partida não encontradas'}), 404
    except ValueError as e:
        return jsonify({'message': str(e)}), 400




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
