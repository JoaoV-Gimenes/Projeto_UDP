## import das bibliotecas ##

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import time

class DetectorMaos:
    """Classe responsável pela detecção das mãos"""
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5,
                 rastreio_confianca=0.5, cor_pontos=(0, 0, 255), cor_conexoes = (255, 255, 255)):

        ## Inicializar Parâmetros
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes
        self.resultado = None
        self.historico = []
        self.janela_frames = 30

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
        if desenho and self.resultado.hand_landmarks and self.resultado:
            for mao in self.resultado.hand_landmarks:
                for ponto in mao:
                    ## Converter coordenadas normmalizadas para pixels
                    h, w, _ = imagem.shape
                    x = int(ponto.x * w)
                    y = int(ponto.y * h)

                    ## desenhar ponto
                    cv2.circle(imagem, (x, y), 5, self.cor_pontos, -1)



def main():
    ##  Capturar o vídeo pela webcam
    cap = cv2.VideoCapture(0)

    ## Instanciar a classe do detector
    detector = DetectorMaos()

    ## Realiza a captura
    while True:
        ## Obtem a imagem
        _,imagem = cap.read()

        ## inverte a imagem para original
        imagem = cv2.flip(imagem,1)

        ## Realização da detecção das mãos
        detector.encontra_mao(imagem)

        ## mostra a imagem de captura
        cv2.imshow('Captura', imagem)

        ## Tempo de atualização da captura
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
