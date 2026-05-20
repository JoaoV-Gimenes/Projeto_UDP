import cv2
import numpy as np
import mediapipe as mp
import os
import jason
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

## Caminho da pasta com os vídeos
PASTA_DATASET = 'dataset'

## Sinais que necessitam de movimento
SINAIS_DINAMICOS = [H, J, K, X, Z]

## Onde salva os dados extraídos
PASTA_SAIDA = 'dados_extraidos'

if os.path.isdir(PASTA_DATASET):
    for sinal in os.listdir(PASTA_DATASET):
        caminho_sinal = os.path.join(PASTA_DATASET, i)

        if os.path.isdir(caminho_sinal):
            for arquivo in os.listdir(caminho_sinal):
                if arquivo.endswith('.mp4'):
                    caminho_video = os.path.join(caminho_sinal, arquivo)
                    cap = cv2.VideoCapture(caminho_video)
                    options = HandLandmarkerOptions


