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
from tensorflow.keras.layers import Input, LSTM, Dense, concatenate
from tensorflow.keras.models import Model

## Adicionar no início do código, após os imports
PASTA_SAIDA = 'dados_extraidos'

## Índices dos pontos principais do rosto
CONTORNO = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LABIOS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OLHO_ESQ = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
OLHO_DIR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
SOBRANCELHA_ESQ = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
SOBRANCELHA_DIR = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
PONTOS_ROSTO = CONTORNO + LABIOS + OLHO_ESQ + OLHO_DIR + SOBRANCELHA_ESQ + SOBRANCELHA_DIR

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

## Quantos pontos do rosto estamos usando
N = len(PONTOS_ROSTO) * 3  ## quantidade de pontos × 3 coordenadas (x, y, z)

## Número de sinais dinâmicos
num_sinais = len(np.unique(y_dinamico))

## Entrada 1 - mao
entrada_mao = Input(shape=(60, 63))     ## 60 frames / 63 valores por frame
lstm_mao = LSTM(64)(entrada_mao)        ## Processa a sequência da mão

## Entrada 2 - rosto
entrada_rosto = Input(shape=(60, N))        ## 60 frames / N valores por frame
lstm_rosto = LSTM(32)(entrada_rosto)        ## Processa a sequência do rosto

## Junta os dois
combinado = concatenate([lstm_mao, lstm_rosto])

## Camadas finais
denso = Dense(64, activation='relu')(combinado)
saida = Dense(num_sinais, activation='softmax')(denso)

## modelo dinãmico
modelo_dinamico = Model(inputs=[entrada_mao, entrada_rosto], outputs=saida)

modelo_dinamico.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo_dinamico.fit(
    [X_treino_dinamico_mao, X_treino_dinamico_rosto],
    y_treino_dinamico,
    epochs=50
)

## Avaliação modelo dinãmico
y_previsao_din = modelo_dinamico.predict([X_teste_dinamico_mao, X_teste_dinamico_rosto])
## Pega o indice com maior probabilidade
y_previsao_din = np.argmax(y_previsao_din, axis=1)
precisao_din = accuracy_score(y_teste_dinamico, y_previsao_din)
print(f'Precisão do modelo dinâmico: {precisao_din * 100:.2f}%')

## Salvamento dos modelos

## Modelo Estático
with open('modelo_estatico.pkl', 'wb') as f:
    pickle.dump(modelo_estatico, f)

with open('le_estatico.pkl', 'wb') as f:
    pickle.dump(le_estatico, f)

## Modelo Dinâmico
modelo_dinamico.save('modelo_dinamico.h5')

with open('le_dinamico.pkl', 'wb') as f:
    pickle.dump(le_dinamico, f)
