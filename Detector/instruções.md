# Detector de Mãos e Rosto

Projeto de detecção de mãos e rosto em tempo real utilizando MediaPipe e OpenCV.

---

## Requisitos

- Python 3.10 ou inferior
- Webcam

---

## Instalação das Bibliotecas

### PowerShell
```powershell
pip install opencv-python
pip install mediapipe
pip install numpy
pip install tensorflow
pip install scikit-learn
```

### Bash
```bash
pip install opencv-python
pip install mediapipe
pip install numpy
pip install tensorflow
pip install scikit-learn
```

## Download dos Modelos

Os modelos de detecção não estão incluídos no repositório e precisam ser baixados uma única vez antes de rodar o projeto.

Coloque os arquivos baixados na **mesma pasta** do `main.py`.

### Hand Landmarker

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile "hand_landmarker.task"
```

**Bash:**
```bash
wget -O hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

---

### Face Landmarker

**PowerShell:**
```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker_v2_with_blendshapes.task"
```

**Bash:**
```bash
wget -O face_landmarker_v2_with_blendshapes.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## Estrutura do Projeto

```
Detector_de_mao/
├── dataset/                          # Vídeos e imagens dos sinais
├── dados_extraidos/                  # Dados extraídos pelo coletar_dados.py
├── main.py                           # Detecção em tempo real
├── coletar_dados.py                  # Extrai pontos dos vídeos e imagens
├── treinar_modelo.py                 # Treina os modelos de reconhecimento
├── modelo_estatico.pkl               # Modelo treinado para letras estáticas
├── modelo_dinamico.h5                # Modelo treinado para sinais dinâmicos
├── hand_landmarker.task              # Modelo de detecção de mãos
├── face_landmarker_v2_with_blendshapes.task  # Modelo de detecção de rosto
├── requirements.txt                  # Bibliotecas necessárias
└── README.md
```

---

## Como Rodar

**Extrair pontos dos vídeos e imagens:**
```bash
python coletar_dados.py
```

**Treinar os modelos:**
```bash
python treinar_modelo.py
```

**Rodar a detecção em tempo real:**
```bash
python main.py
```

---

## Funcionalidades

- Detecção de mãos em tempo real com 21 pontos de referência
- Detecção de rosto com malha facial, contornos e íris
- Detecção de expressões faciais (blendshapes)
- Histórico de movimentos para reconhecimento de gestos dinâmicos
- Reconhecimento de sinais estáticos (letras do alfabeto)
- Reconhecimento de sinais dinâmicos (palavras com movimento)