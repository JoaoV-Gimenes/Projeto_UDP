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
```

### Bash
```bash
pip install opencv-python
pip install mediapipe
pip install numpy
```

---

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
├── main.py
├── hand_landmarker.task
├── face_landmarker_v2_with_blendshapes.task
└── README.md
```

---

## Como Rodar

```bash
python main.py
```

---

## Funcionalidades

- Detecção de mãos em tempo real com 21 pontos de referência
- Detecção de rosto com malha facial, contornos e íris
- Detecção de expressões faciais (blendshapes)
- Histórico de movimentos para reconhecimento de gestos dinâmicos
    
        