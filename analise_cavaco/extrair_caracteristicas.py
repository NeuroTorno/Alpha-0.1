import cv2
import numpy as np
import os

def calcular_percentual(imagem, cor_baixo, cor_alto):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, cor_baixo, cor_alto)
    percentual = (np.sum(mascara > 0) / mascara.size) * 100
    return percentual

def extrair_dados(diretorio):
    dados = []
    labels = []

    for categoria in ['prateado', 'dourado', 'queimado']:
        caminho_categoria = os.path.join(diretorio, categoria)
        for arquivo in os.listdir(caminho_categoria):
            caminho_imagem = os.path.join(caminho_categoria, arquivo)
            imagem = cv2.imread(caminho_imagem)
            if imagem is not None:
                azul = calcular_percentual(imagem, np.array([90, 50, 50]), np.array([140, 255, 255]))
                dourado = calcular_percentual(imagem, np.array([20, 100, 100]), np.array([40, 255, 255]))
                prateado = calcular_percentual(imagem, np.array([0, 0, 150]), np.array([180, 50, 255]))
                
                dados.append([azul, dourado, prateado])
                
                if categoria == 'prateado':
                    labels.append(0)
                elif categoria == 'dourado':
                    labels.append(1)
                else:  # queimado
                    labels.append(2)

    return np.array(dados), np.array(labels)

if __name__ == "__main__":
    X, y = extrair_dados('cavacos')
    np.save('dados_X.npy', X)
    np.save('dados_y.npy', y)
    print('[INFO] Extração de características concluída.')
