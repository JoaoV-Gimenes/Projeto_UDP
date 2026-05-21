import cv2
import numpy as np
import mediapipe as mp
import os
import json
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

## Caminho da pasta com os vídeos
PASTA_DATASET = 'dataset'

## Sinais que necessitam de movimento
SINAIS_DINAMICOS = ['H', 'J', 'K', 'X', 'Z']

## Onde salva os dados extraídos
PASTA_SAIDA = 'dados_extraidos'

## Config e detector para vídeo
options_video = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )
detector_video = HandLandmarker.create_from_options(options_video)


## Config e detector para imagem
options_estatico = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    )
detector_imagem = HandLandmarker.create_from_options(options_estatico)

if os.path.isdir(PASTA_DATASET):
    for sinal in os.listdir(PASTA_DATASET):
        caminho_sinal = os.path.join(PASTA_DATASET, sinal)

        if os.path.isdir(caminho_sinal):
            for arquivo in os.listdir(caminho_sinal):

                ## Caso for vídeo
                if arquivo.endswith('.mp4'):
                    caminho_video = os.path.join(caminho_sinal, arquivo)
                    cap = cv2.VideoCapture(caminho_video)

                    frames = []
                    while True:
                        ## verific diz se algo foi obtido ou não (True/False)
                        verific, frame = cap.read()

                        ## Caso não tenha mais frames no vídeo, Para o ciclo
                        if not verific:
                            break

                        ## Processamento
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_rgb = np.array(frame_rgb)
                        mp_frame = mp.Image(image_format= mp.ImageFormat.SRGB, data=frame_rgb )

                        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                        resultado = detector_video.detect_for_video(mp_frame, timestamp)

                        if resultado.hand_landmarks:
                            pontos = []
                            for mao in resultado.hand_landmarks:
                                for ponto in mao:
                                    x = ponto.x
                                    y = ponto.y
                                    z = ponto.z
                                    pontos.append([x, y, z])

                            ## Caso for um vídeo de um sinal dinâmico ou não
                            if sinal in SINAIS_DINAMICOS:
                                frames.append(pontos)
                            else:
                                frames = pontos
                                break

                    ## Libera o vídeo para JSON
                    cap.release()

                    ## como será salvo
                    if sinal in SINAIS_DINAMICOS:
                        dados = {
                            'sinal': sinal,
                            'tipo': 'dinamico',
                            'frames': frames
                        }
                    else:
                        dados = {
                            'sinal': sinal,
                            'tipo': 'estatico',
                            'pontos': frames
                        }

                    ## Cria uma pasta do sinal dentro de PASTA_SAIDA caso não tenha
                    pasta_saida_sinal = os.path.join(PASTA_SAIDA, sinal)
                    os.makedirs(pasta_saida_sinal, exist_ok=True)

                    ## Substitui o tipo do arquivo para .json
                    nome_arquivo = arquivo.replace('.mp4', '.json')

                    ## Coloca o arquivo e .json na pasta_saida_sinal
                    caminho_saida = os.path.join(pasta_saida_sinal, nome_arquivo)

                    with open(caminho_saida, 'w') as f:
                        json.dump(dados, f)

                    print(f'✓ {sinal} - {arquivo} processado!')

                ## Caso for imagem
                elif arquivo.endswith('.png'):

                    ## pega o caminho da imagem e a lê
                    caminho_imagem = os.path.join(caminho_sinal, arquivo)
                    imagem = cv2.imread(caminho_imagem)

                    ## BRG para RGB e muda para arrays numpy
                    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                    imagem_rgb = np.array(imagem_rgb)

                    ## Muda para o formato aceito pelo mediapipe
                    mp_imagem = mp.Image(image_format= mp.ImageFormat.SRGB, data=imagem_rgb )

                    ## salva o resultado com os pontos detectados nas mãos
                    resultado = detector_imagem.detect(mp_imagem)

                    if resultado.hand_landmarks:
                        ## Salva a coordenada de cada ponto
                        pontos = []
                        for mao in resultado.hand_landmarks:
                            for ponto in mao:
                                x = ponto.x
                                y = ponto.y
                                z = ponto.z
                                pontos.append([x, y, z])

                        dados = {
                            'sinal': sinal,
                            'tipo': 'estatico',
                            'pontos': pontos
                        }

                        ## Cria uma pasta do sinal dentro de PASTA_SAIDA caso não tenha
                        pasta_saida_sinal = os.path.join(PASTA_SAIDA, sinal)
                        os.makedirs(pasta_saida_sinal, exist_ok=True)

                        ## Substitui o tipo do arquivo para .json
                        nome_arquivo = arquivo.replace('.png', '.json')

                        ## Coloca o arquivo e .json na pasta_saida_sinal
                        caminho_saida = os.path.join(pasta_saida_sinal, nome_arquivo)

                        with open(caminho_saida, 'w') as f:
                            json.dump(dados, f)

                        print(f'✓ {sinal} - {arquivo} processado!')

print('Extração concluída!')