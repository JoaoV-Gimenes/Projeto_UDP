import cv2
import numpy as np
import mediapipe as mp
import os
import json
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode, FaceLandmarker, FaceLandmarkerOptions

## Caminho da pasta com os vídeos
PASTA_DATASET = 'dataset'

## Sinais que necessitam de movimento
SINAIS_DINAMICOS = ['H', 'J', 'K', 'X', 'Z']

## Onde salva os dados extraídos
PASTA_SAIDA = 'dados_extraidos'

## Índices dos pontos principais do rosto
CONTORNO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LABIOS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OLHO_ESQ = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
OLHO_DIR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
SOBRANCELHA_ESQ = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
SOBRANCELHA_DIR = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

## Lista única com todos os pontos do rosto
PONTOS_ROSTO = CONTORNO + LABIOS + OLHO_ESQ + OLHO_DIR + SOBRANCELHA_ESQ + SOBRANCELHA_DIR

## Config e detector para vídeo -- MAOS
options_video_mao = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )
detector_video_mao = HandLandmarker.create_from_options(options_video_mao)


## Config e detector para imagem -- MAOS
options_estatico_mao = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    )
detector_imagem_mao = HandLandmarker.create_from_options(options_estatico_mao)

## Config e detector para video -- ROSTO
options_dinamico_rosto = FaceLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path = 'face_landmarker_v2_with_blendshapes.task'),
            running_mode = RunningMode.VIDEO,
            num_faces = 1,
            min_face_detection_confidence = 0.5
        )
detector_video_rosto = FaceLandmarker.create_from_options(options_dinamico_rosto)

## Config e detector para imagem -- ROSTO
options_estatico_rosto = FaceLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path = 'face_landmarker_v2_with_blendshapes.task'),
            running_mode = RunningMode.IMAGE,
            num_faces = 1,
            min_face_detection_confidence = 0.5
        )
detector_imagem_rosto = FaceLandmarker.create_from_options(options_estatico_rosto)

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
                        resultado = detector_video_mao.detect_for_video(mp_frame, timestamp)

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
                    resultado = detector_imagem_mao.detect(mp_imagem)

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