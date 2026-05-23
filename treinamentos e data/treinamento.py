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

## Listas para armazenar os pontos de cada imagem/frame

X_estatico =[]
y_estatico =[]

###
X_dinamico =[]
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
                        X_dinamico.append(dados['frames'])
                        y_dinamico.append(dados['sinal'])

                    else:
                        X_estatico.append(dados['pontos'])
                        y_estatico.append(dados['sinal'])

## Todos para array numpy (scikit-learn e o tensorflow trabalham com arrays numpy)
X_estatico = np.array(X_estatico)
y_estatico = np.array(y_estatico)
X_dinamico = np.array(X_dinamico)
y_dinamico = np.array(y_dinamico)

## Letras para números
le_estatico = LabelEncoder()
y_estatico = le_estatico.fit_transform(y_estatico)

le_dinamico = LabelEncoder()
y_dinamico = le_dinamico.fit_transform(y_dinamico)

## Separação de treino e teste (80, 20)
X_treino_estatico, X_teste_estatico, y_treino_estatico, y_teste_estatico = train_test_split(X_estatico, y_estatico, test_size = 0.2, random_state = 42)

X_treino_dinamico, X_teste_dinamico, y_treino_dinamico, y_teste_dinamico = train_test_split(X_dinamico, y_dinamico, test_size = 0.2, random_state = 42)

## Treino modelo random Forest
modelo_estatico = RandomForestClassifier(n_estimators=100)
modelo_estatico.fit(X_treino_estatico, y_treino_estatico)

## Avaliação do teste
y_previsao = modelo_estatico.predict(X_teste_estatico)
precisao = accuracy_score(y_teste_estatico, y_previsao)
print(f'Precisão do modelo estático: {precisao * 100:.2f}%')

