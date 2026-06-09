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

## Detectores exclusivos para vídeo espelhado
options_esp_mao = HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
detector_esp_mao = HandLandmarker.create_from_options(options_esp_mao)

options_esp_rosto = FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task'),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5
)
detector_esp_rosto = FaceLandmarker.create_from_options(options_esp_rosto)

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

                        ## timestamp em que os frames serão analisados
                        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

                        ## salva o resultado com os pontos detectados nas mãos/rosto
                        resultado_mao = detector_video_mao.detect_for_video(mp_frame, timestamp)
                        resultado_rosto = detector_video_rosto.detect_for_video(mp_frame, timestamp)

                        if resultado_mao.hand_landmarks:
                            ## Salva a coordenada de cada ponto na mão
                            pontos_mao = []
                            for mao in resultado_mao.hand_landmarks:
                                for ponto in mao:
                                    x = ponto.x
                                    y = ponto.y
                                    z = ponto.z
                                    pontos_mao.append([x, y, z])

                            pontos_rosto = []
                            if resultado_rosto.face_landmarks:
                                for rosto in resultado_rosto.face_landmarks:
                                    for i in PONTOS_ROSTO:
                                        ponto = rosto[i]
                                        pontos_rosto.append([ponto.x, ponto.y, ponto.z])

                            ## Caso for um vídeo de um sinal dinâmico ou não
                            if sinal in SINAIS_DINAMICOS:
                                frames.append({
                                    'mao': pontos_mao,
                                    'rosto': pontos_rosto
                                })
                            else:
                                frames = {
                                    'mao': pontos_mao,
                                    'rosto': pontos_rosto
                                }
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

                    ## Processaento video espelhado

                    cap_esp = cv2.VideoCapture(caminho_video)
                    frames_esp = []

                    while True:
                        verific, frames = cap_esp.read()
                        if not verific:
                            break

                        ## Espelhar o frame
                        frame_esp = cv2.flip(frames, 1)

                        frame_esp_rgb = cv2.cvtColor(frame_esp, cv2.COLOR_BGR2RGB)
                        frame_esp_rgb = np.array(frame_esp_rgb)
                        mp_frame_esp = mp.Image(image_format= mp.ImageFormat.SRGB, data=frame_esp_rgb)

                        timestamp_esp = int(cap_esp.get(cv2.CAP_PROP_POS_MSEC))

                        resultado_mao = detector_esp_mao.detect_for_video(mp_frame_esp, timestamp_esp)
                        resultado_rosto = detector_esp_rosto.detect_for_video(mp_frame_esp, timestamp_esp)

                        if resultado_mao.hand_landmarks:
                            ## Salva a coordenada de cada ponto na mão
                            pontos_mao = []
                            for mao in resultado_mao.hand_landmarks:
                                for ponto in mao:
                                    x = ponto.x
                                    y = ponto.y
                                    z = ponto.z
                                    pontos_mao.append([x, y, z])

                            pontos_rosto = []
                            if resultado_rosto.face_landmarks:
                                for rosto in resultado_rosto.face_landmarks:
                                    for i in PONTOS_ROSTO:
                                        ponto = rosto[i]
                                        pontos_rosto.append([ponto.x, ponto.y, ponto.z])

                            ## Caso for um vídeo de um sinal dinâmico ou não
                            if sinal in SINAIS_DINAMICOS:
                                frames_esp.append({
                                    'mao': pontos_mao,
                                    'rosto': pontos_rosto
                                })
                            else:
                                frames_esp = {
                                    'mao': pontos_mao,
                                    'rosto': pontos_rosto
                                }
                                break

                    ## Libera o vídeo para JSON
                    cap_esp.release()

                    ## como será salvo
                    if sinal in SINAIS_DINAMICOS:
                        dados = {
                            'sinal': sinal,
                            'tipo': 'dinamico',
                            'frames': frames_esp
                        }
                    else:
                        dados = {
                            'sinal': sinal,
                            'tipo': 'estatico',
                            'pontos': frames_esp
                        }

                    ## Cria uma pasta do sinal dentro de PASTA_SAIDA caso não tenha
                    pasta_saida_sinal = os.path.join(PASTA_SAIDA, sinal)
                    os.makedirs(pasta_saida_sinal, exist_ok=True)

                    ## Substitui o tipo do arquivo para .json
                    nome_arquivo = arquivo.replace('.mp4', '_espelhado.json')

                    ## Coloca o arquivo e .json na pasta_saida_sinal
                    caminho_saida = os.path.join(pasta_saida_sinal, nome_arquivo)

                    with open(caminho_saida, 'w') as f:
                        json.dump(dados, f)

                    print(f'✓ {sinal} - {arquivo} espelhado processado!')

                ## Caso for imagem
                elif arquivo.endswith('.png') or arquivo.endswith('.jpg'):

                    ## pega o caminho da imagem e a lê
                    caminho_imagem = os.path.join(caminho_sinal, arquivo)
                    imagem = cv2.imread(caminho_imagem)

                    ## BRG para RGB e muda para arrays numpy
                    imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
                    imagem_rgb = np.array(imagem_rgb)

                    ## Muda para o formato aceito pelo mediapipe
                    mp_imagem = mp.Image(image_format= mp.ImageFormat.SRGB, data=imagem_rgb )

                    ## salva o resultado com os pontos detectados nas mãos/rosto
                    resultado_mao = detector_imagem_mao.detect(mp_imagem)
                    resultado_rosto = detector_imagem_rosto.detect(mp_imagem)

                    if resultado_mao.hand_landmarks:
                        ## Salva a coordenada de cada ponto na mão
                        pontos_mao = []
                        for mao in resultado_mao.hand_landmarks:
                            for ponto in mao:
                                x = ponto.x
                                y = ponto.y
                                z = ponto.z
                                pontos_mao.append([x, y, z])

                        ## Salva a coordenada de cada ponto no rosto
                        pontos_rosto = []
                        if resultado_rosto.face_landmarks:
                            for rosto in resultado_rosto.face_landmarks:
                                for i in PONTOS_ROSTO:
                                    ponto = rosto[i]
                                    pontos_rosto.append([ponto.x, ponto.y, ponto.z])

                        dados = {
                            'sinal': sinal,
                            'tipo': 'estatico',
                            'pontos': {
                                'mao': pontos_mao,
                                'rosto': pontos_rosto ## Lista vazia caso não encontre rosto
                            }
                        }

                        ## Cria uma pasta do sinal dentro de PASTA_SAIDA caso não tenha
                        pasta_saida_sinal = os.path.join(PASTA_SAIDA, sinal)
                        os.makedirs(pasta_saida_sinal, exist_ok=True)

                        ## Substitui o tipo do arquivo para .json
                        nome_arquivo_png = arquivo.replace('.png', '.json').replace('.jpg', '.json')

                        ## Coloca o arquivo e .json na pasta_saida_sinal
                        caminho_saida_png = os.path.join(pasta_saida_sinal, nome_arquivo_png)

                        ## Salva os dados
                        with open(caminho_saida_png, 'w') as f:
                            json.dump(dados, f)

                        print(f'✓ {sinal} - {arquivo} processado!')

                    ###### Processamento da imagem espelhada (para mão esquerda)
                    ## pega o caminho da imagem e a lê
                    caminho_imagem = os.path.join(caminho_sinal, arquivo)
                    imagem = cv2.imread(caminho_imagem)

                    ## Espelha a imagem original
                    imagem_espelhada = cv2.flip(imagem, 1)

                    ## BRG para RGB e muda para arrays numpy
                    imagem_esp_rgb = cv2.cvtColor(imagem_espelhada, cv2.COLOR_BGR2RGB)
                    imagem_esp_rgb = np.array(imagem_esp_rgb)

                    ## Muda para o formato aceito pelo mediapipe
                    mp_imagem_esp = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_esp_rgb)

                    ## salva o resultado com os pontos detectados nas mãos/rosto
                    resultado_mao = detector_imagem_mao.detect(mp_imagem_esp)
                    resultado_rosto = detector_imagem_rosto.detect(mp_imagem_esp)

                    if resultado_mao.hand_landmarks:
                        ## Salva a coordenada de cada ponto na mão
                        pontos_mao = []
                        for mao in resultado_mao.hand_landmarks:
                            for ponto in mao:
                                x = ponto.x
                                y = ponto.y
                                z = ponto.z
                                pontos_mao.append([x, y, z])

                        ## Salva a coordenada de cada ponto no rosto
                        pontos_rosto = []
                        if resultado_rosto.face_landmarks:
                            for rosto in resultado_rosto.face_landmarks:
                                for i in PONTOS_ROSTO:
                                    ponto = rosto[i]
                                    pontos_rosto.append([ponto.x, ponto.y, ponto.z])

                        dados = {
                            'sinal': sinal,
                            'tipo': 'estatico',
                            'pontos': {
                                'mao': pontos_mao,
                                'rosto': pontos_rosto  ## Lista vazia caso não encontre rosto
                            }
                        }

                        ## Cria uma pasta do sinal dentro de PASTA_SAIDA caso não tenha
                        pasta_saida_sinal = os.path.join(PASTA_SAIDA, sinal)
                        os.makedirs(pasta_saida_sinal, exist_ok=True)

                        ## Substitui o tipo do arquivo para .json
                        nome_arquivo = arquivo.replace('.png', '_espelhado.json')

                        ## Coloca o arquivo e .json na pasta_saida_sinal
                        caminho_saida = os.path.join(pasta_saida_sinal, nome_arquivo)

                        with open(caminho_saida, 'w') as f:
                            json.dump(dados, f)

                        print(f'✓ {sinal} - {arquivo} espelhado processado!')

print('Extração concluída!')