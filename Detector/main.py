## import das bibliotecas ##

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode, drawing_utils, drawing_styles, FaceLandmarker, FaceLandmarkerOptions
import time
import pickle
from tensorflow.keras.models import load_model


class DetectorRosto:
    """Classe responsável pela detecção do rosto"""

    def __init__(self, max_rosto= 1, deteccao_confianca= 0.5):

        self.resultado = None
        self.blendshapes = []

        options = FaceLandmarkerOptions(
            base_options = python.BaseOptions(model_asset_path = 'face_landmarker_v2_with_blendshapes.task'),
            running_mode = RunningMode.LIVE_STREAM,
            num_faces = max_rosto,
            min_face_detection_confidence = deteccao_confianca,
            output_face_blendshapes = True,
            result_callback = self.callback_resultado
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def callback_resultado (self, resultado, imagem_saida, timestamp_ms):
        self.resultado = resultado
        self.blendshapes = resultado.face_blendshapes if resultado.face_blendshapes else []

    def encontra_rosto(self, imagem, desenho = True):

        ## BGR para RGB
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem_rgb = np.array(imagem_rgb)

        ## Converte para o formato do mediapipe
        mp_imagem = mp.Image(image_format=mp.ImageFormat.SRGB, data = imagem_rgb)

        ## Timestamp
        timestamp = int(time.time()*1000)

        ## Envia para a detecção assíncrona
        self.detector.detect_async(mp_imagem, timestamp)

        ## Desenha os landmarks
        if desenho and self.resultado and self.resultado.face_landmarks:
            for rosto in self.resultado.face_landmarks:
                ## Malha do rosto
                drawing_utils.draw_landmarks(
                    image=imagem,
                    landmark_list=rosto,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())

                ## Contornos
                drawing_utils.draw_landmarks(
                    image=imagem,
                    landmark_list=rosto,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

                ## Íris esquerda
                drawing_utils.draw_landmarks(
                    image=imagem,
                    landmark_list=rosto,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

                ## Íris direita
                drawing_utils.draw_landmarks(
                    image=imagem,
                    landmark_list=rosto,
                    connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())

    def detecta_gestos(self, confianca_minima):
        if not self.blendshapes:
            return []
        gestos_detectados = []
        for blendshape in self.blendshapes[0]:
            if blendshape.score >= confianca_minima:
                gestos_detectados.append({
                    'gesto': blendshape.category_name,
                    'confianca' : round(blendshape.score, 2)
                })
        return gestos_detectados






class DetectorMaos:
    """Classe responsável pela detecção das mãos"""
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5,
                 rastreio_confianca=0.5, cor_pontos=(0, 0, 255), cor_conexoes = (255, 255, 255)):

        ## Inicializar Parâmetros
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes
        self.resultado = None
        self.historico = []
        self.janela_frames = 60

        ## Configurar o HandLandmarker
        options = HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=max_maos,
            min_hand_detection_confidence=deteccao_confianca,
            min_tracking_confidence=rastreio_confianca,
            result_callback=self.callback_resultado
        )

        self.detector = HandLandmarker.create_from_options(options)

        ## Configurações do desenho dos pontos
        self.desenho_config_pontos = self.cor_pontos

        ## Configurações do desenho das conexões
        self.desenho_config_conexoes = self.cor_conexoes

    def callback_resultado(self, resultado, imagem_saida, timestamp_ms):
        """Função chamada automáticamente quando a detecção termina"""
        self.resultado = resultado

        ## Guardar histórico para gestos dinâmicos
        if resultado.hand_landmarks:
            for mao in resultado.hand_landmarks:
                pontos = [(p.x, p.y, p.z) for p in mao]
                self.historico.append({
                    'timestamp': timestamp_ms,
                    'pontos': pontos
                })
                ## Manter apenas os últimos N frames
                if len(self.historico) > self.janela_frames:
                    self.historico.pop(0)


    def encontra_mao(self, imagem, desenho=True):
        """Função responsável por detectar a(s) mao(s)"""

        ## Conversão BGR para RGB
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        imagem_rgb = np.array(imagem_rgb)

        ## Converter para o formato aceito pelo mediapipe
        mp_imagem = mp.Image(image_format = mp.ImageFormat.SRGB, data=imagem_rgb)

        ## Pegar o timestamp em Milissegundo
        timestamp = int(time.time()*1000)

        ## Enviar para detecção assíncrona
        self.detector.detect_async(mp_imagem, timestamp)

        ## Desenhar os pontos se encontrar mãos e desenho = True
        if desenho and self.resultado and self.resultado.hand_landmarks:
            for mao in self.resultado.hand_landmarks:
                for ponto in mao:
                    ## Converter coordenadas normmalizadas para pixels
                    h, w, _ = imagem.shape
                    x = int(ponto.x * w)
                    y = int(ponto.y * h)

                    ## desenhar ponto
                    cv2.circle(imagem, (x, y), 5, self.cor_pontos, -1)

## Índices dos pontos principais do rosto
CONTORNO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LABIOS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OLHO_ESQ = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
OLHO_DIR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
SOBRANCELHA_ESQ = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
SOBRANCELHA_DIR = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
PONTOS_ROSTO = CONTORNO + LABIOS + OLHO_ESQ + OLHO_DIR + SOBRANCELHA_ESQ + SOBRANCELHA_DIR

## Carrega os modelos
with open('modelo_estatico.pkl', 'rb') as f:
    modelo_estatico = pickle.load(f)

with open('le_estatico.pkl', 'rb') as f:
    le_estatico = pickle.load(f)

with open('le_dinamico.pkl', 'rb') as f:
    le_dinamico = pickle.load(f)

modelo_dinamico = load_model('modelo_dinamico.h5')

def main():
    ##  Capturar o vídeo pela webcam
    cap = cv2.VideoCapture(0)

    ## Instanciar a classes
    detector = DetectorMaos()
    detector_rosto = DetectorRosto()

    ## Realiza a captura
    while True:
        ## Obtem a imagem
        _,imagem = cap.read()

        ## inverte a imagem para original
        imagem = cv2.flip(imagem,1)

        ## Realização da detecção das mãos
        detector.encontra_mao(imagem)
        detector_rosto.encontra_rosto(imagem)

        ## Salva os pontos para análise
        pontos_mao = []
        if detector.resultado and detector.resultado.hand_landmarks:
            for mao in detector.resultado.hand_landmarks:
                for ponto in mao:
                    pontos_mao.append([ponto.x, ponto.y, ponto.z])

        pontos_rosto = []
        if detector_rosto.resultado and detector_rosto.resultado.face_landmarks:
            for rosto in detector_rosto.resultado.face_landmarks:
                for i in PONTOS_ROSTO:
                    ponto = rosto[i]
                    pontos_rosto.append([ponto.x, ponto.y, ponto.z])

        ## Só classifica se detectou mão
        if pontos_mao:
            if len(detector.historico) >= 60:
                ## Pega os últimos 60 frames do histórico
                sequencia_mao = [frames['pontos'] for frames in detector.historico[-60:]]
                sequencia_mao = np.array([sequencia_mao]) ## Formato (1, 60, 63)

                ## rosto do frame atual
                sequencia_rosto = np.array([pontos_rosto]* 60) ## repete o rosto atual 60 vezes
                sequencia_rosto = sequencia_rosto.reshape(1, 60, -1) ## Formato (1, 60, N)

                ## Passar para o modelo
                previsao = modelo_dinamico.predict([sequencia_mao, sequencia_rosto])
                indice = np.argmax(previsao)
                letra = le_dinamico.inverse_transform([indice])[0]
                cv2.putText(imagem, letra, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                entrada = np.array(pontos_mao + pontos_rosto).flatten()
                entrada = entrada.reshape(1, -1)
                previsao = modelo_estatico.predict(entrada)
                letra = le_estatico.inverse_transform(previsao)[0]
                cv2.putText(imagem, letra, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        ## mostra a imagem de captura
        cv2.imshow('Captura', imagem)

        ## Tempo de atualização da captura
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
