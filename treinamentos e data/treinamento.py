## import das bibliotecas
import json
import os
import numpy as np
import tensorflow as tf
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

## Listas para armazenar os pontos de cada imagem/frame

X_estatico_mao =[]
X_estatico_rosto =[]
y_estatico =[]

###
X_dinamico_mao =[]
X_dinamico_rosto =[]
y_dinamico =[]


if os.path.isdir(PASTA_SAIDA):

    ## Acessa cada subpasta de letras
    for letra in os.listdir(PASTA_SAIDA):

        ## Cria o caminho para as subpastas
        caminho_letra = os.path.join(PASTA_SAIDA, letra)

        if os.path.isdir(caminho_letra):
            ## Acessa cada arquivo dentro das subpastas das letras
            for arquivo in os.listdir(caminho_letra):

                ## Cria o caminho para os arquivos
                caminho_arquivo = os.path.join(caminho_letra, arquivo)

                ## lê cada arquivo .json
                with open(caminho_arquivo, 'r') as f:
                    dados = json.load(f)

                    ## Separa dinamico e estático
                    if dados['tipo'] == 'dinamico':
                        ## Extrair mao e rosto de cada frame
                        mao_frames = [frame['mao'] for frame in dados['frames']]
                        rosto_frames = [frame['rosto'] for frame in dados['frames']]

                        X_dinamico_mao.append(mao_frames)
                        X_dinamico_rosto.append(rosto_frames)
                        y_dinamico.append(dados['sinal'])

                    else:
                        X_estatico_mao.append(dados['pontos']['mao'])
                        X_estatico_rosto.append(dados['pontos']['rosto'])
                        y_estatico.append(dados['sinal'])

## Todos para array numpy (scikit-learn e o tensorflow trabalham com arrays numpy)
X_estatico_rosto = np.array(X_estatico_rosto)
X_estatico_mao = np.array(X_estatico_mao)
y_estatico = np.array(y_estatico)
X_dinamico_rosto = np.array(X_dinamico_rosto)
X_dinamico_mao = np.array(X_dinamico_mao)
y_dinamico = np.array(y_dinamico)

## Letras para números
le_estatico = LabelEncoder()
y_estatico = le_estatico.fit_transform(y_estatico)

le_dinamico = LabelEncoder()
y_dinamico = le_dinamico.fit_transform(y_dinamico)

## Separação de treino e teste (80, 20)
X_treino_estatico_mao, X_teste_estatico_mao, X_treino_estatico_rosto, X_teste_estatico_rosto, y_treino_estatico, y_teste_estatico = train_test_split(X_estatico_mao, X_estatico_rosto, y_estatico, test_size = 0.2, random_state = 42)

X_treino_dinamico_mao, X_teste_dinamico_mao, X_treino_dinamico_rosto, X_teste_dinamico_rosto,  y_treino_dinamico, y_teste_dinamico = train_test_split(X_dinamico_mao, X_dinamico_rosto, y_dinamico, test_size = 0.2, random_state = 42)

X_treino_estatico = np.hstack([X_treino_estatico_mao, X_treino_estatico_rosto])
X_teste_estatico = np.hstack([X_teste_estatico_mao, X_teste_estatico_rosto])

## Treino modelo random Forest
modelo_estatico = RandomForestClassifier(n_estimators=100)
modelo_estatico.fit(X_treino_estatico, y_treino_estatico)

## Avaliação do teste
y_previsao = modelo_estatico.predict(X_teste_estatico)
precisao = accuracy_score(y_teste_estatico, y_previsao)
print(f'Precisão do modelo estático: {precisao * 100:.2f}%')

X_treino_dinamico_mao = pad_sequences(X_treino_dinamico_mao, maxlen=60, padding='pre', truncating='pre', dtype='float32')
X_teste_dinamico_mao = pad_sequences(X_teste_dinamico_mao, maxlen= 60, padding='pre', truncating='pre', dtype='float32')

X_treino_dinamico_rosto = pad_sequences(X_treino_dinamico_rosto, maxlen=60, padding='pre', truncating='pre', dtype='float32')
X_teste_dinamico_rosto = pad_sequences(X_teste_dinamico_rosto, maxlen=60, padding='pre', truncating='pre', dtype='float32')
