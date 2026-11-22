# Instaplay

Sistema de detecção de gestos em tempo real para jogos multiplayer. Quando um jogador faz um **X com os antebraços** por 2 segundos, o sistema salva automaticamente os últimos 30 segundos de vídeo com marca d'água e registra no banco de dados.

## Como funciona

1. A câmera captura o vídeo continuamente em uma thread dedicada
2. O modelo **RTMO** detecta a pose de todas as pessoas no frame em outra thread
3. Quando alguém mantém **o X com os antebraços por 2 segundos**, o gesto é confirmado
4. O sistema toca um som, salva o clipe e registra no banco
5. A API Flask permite buscar os vídeos pelo código da partida

## Estrutura do projeto

```
instaplay/
├── app.py                      # Loop principal + orquestração das threads
├── config.py                   # Constantes (FPS, buffer, resolução, paths)
├── camera/
│   ├── camera.py               # Captura de vídeo em thread dedicada
│   ├── detector.py             # Inferência RTMO em thread dedicada
│   └── pose_utils.py           # Regra do gesto X
├── processing/
│   ├── bufferManager.py        # Buffer circular de 30 segundos
│   └── videoSaver.py           # Exporta vídeo do buffer com marca d'água
├── server/
│   ├── server.py               # API Flask
│   ├── db/connection.py        # Conexão PostgreSQL + modelos ORM
│   └── services/
│       └── partidas_service.py # Regras de negócio (partidas e vídeos)
├── static/
│   ├── logo.png                # Marca d'água aplicada nos vídeos
│   ├── start.mp3               # Som ao iniciar a partida
│   ├── replay.mp3              # Som ao detectar o gesto
│   └── videos/                 # Vídeos gerados
├── docker-compose.yaml         # PostgreSQL
├── init.sql                    # Schema do banco
└── requirements.txt
```

## Pré-requisitos

- Python 3.10+
- Docker Desktop

## Instalação

```bash
# 1. Subir o banco de dados
docker-compose up -d

# 2. Instalar dependências Python
pip install -r requirements.txt
```

Na primeira execução o modelo RTMO-s (~35 MB) é baixado automaticamente para `~/.cache/rtmlib`.

## Modelo de pose

O sistema usa o **RTMO** ([artigo, CVPR 2024](https://arxiv.org/abs/2312.07526)), da OpenMMLab, executado pelo [`rtmlib`](https://github.com/Tau-J/rtmlib) com ONNX Runtime (sem PyTorch).

- **Uma única etapa**: detecta todas as pessoas numa só passada, então o custo quase não muda com o número de jogadores
- **17 pontos no formato COCO**, os mesmos que a regra do gesto usa
- Configurável em `config.py`: `POSE_MODEL` (`rtmo-s`, `rtmo-m` ou `rtmo-l`) e `POSE_RUNTIME` (`onnxruntime` ou `openvino`)

**Por que RTMO e não YOLOv8-pose:** num teste com o mesmo vídeo de webcam, o YOLOv8 (nano e small) perdia um dos braços quando os antebraços se cruzavam e não detectou nenhum gesto X; o RTMO detectou os dois braços e disparou o replay. A comparação foi feita com o projeto [pose-benchmark](../pose-benchmark).

**Licença:** o código do RTMO e do `rtmlib` é Apache-2.0 (permite uso comercial). Os pesos pré-treinados usados foram treinados no conjunto "body7" (7 datasets combinados), e alguns desses datasets podem ter licença apenas para pesquisa. **Antes de comercializar, confirme a licença dos pesos** ou use pesos treinados só no COCO.

## Configuração da câmera

Em `app.py`, linha 43, altere o `source` conforme sua câmera:

| Fonte | Valor |
|---|---|
| Webcam local | `0` |
| IP Cam (DroidCam, iVCam) | `'http://192.168.x.x:4747/video'` |
| RTSP | `'rtsp://usuario:senha@192.168.x.x:554/stream'` |

## Execução

```bash
python app.py
```

O terminal exibe o código da partida ao iniciar:

```
Partida iniciada | Codigo: a3f2c1d4e5b6...
```

O mesmo código aparece na janela de preview. Para encerrar, pressione `q` na janela.

## Detecção de gesto

- Gesto: **X com os antebraços** (na frente do peito ou acima da cabeça)
- Keypoints utilizados: ombros (5, 6), cotovelos (7, 8) e pulsos (9, 10) — formato COCO
- Condições (todas ao mesmo tempo):
  - pulsos cruzados: cada pulso está do lado oposto ao seu ombro
  - cotovelos **não** cruzados (assim os antebraços se cruzam formando o X)
  - pulsos acima dos cotovelos (`pulso.y < cotovelo.y`)
- Funciona com o jogador de frente ou de costas para a câmera; jogadores de lado são ignorados
- Tempo mínimo sustentado: **2 segundos**
- Cooldown entre detecções: **5 segundos**

## API

Base URL: `http://localhost:5000`

### Buscar partida e vídeos

```
GET /partida/<codigo>
```

Resposta:
```json
{
  "id": "uuid",
  "codigo": "a3f2c1d4...",
  "pagamento": false,
  "data_inicio": "2025-01-01T10:00:00",
  "data_fim": null,
  "jogadas": [
    {
      "id": "uuid",
      "partida_id": "uuid",
      "path": "20250101100500-uuid.mp4",
      "thumbnail": "base64...",
      "created_at": "2025-01-01T10:05:00"
    }
  ]
}
```

### Baixar vídeo

```
GET /partida/video/<filename>
```

## Banco de dados

| Tabela | Descrição |
|---|---|
| `partida` | Sessão de jogo (código, data início/fim, pagamento) |
| `partida_videos` | Clipes gerados (path, thumbnail base64, timestamp) |

## Performance

| Técnica | Efeito |
|---|---|
| `rtmo-s` via ONNX Runtime | Modelo mais leve da família RTMO, sem PyTorch |
| Resize para 640px antes da inferência | Reduz custo ~4× vs resolução original |
| `skip_frames=2` | Inferência a cada 2 frames |
| Thread de captura separada | Câmera nunca bloqueia o loop principal |
| Fila de 1 slot no detector | Sempre processa o frame mais recente |
