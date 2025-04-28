import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Carrega o modelo treinado
with open('modelo_svm.pkl', 'rb') as f:
    modelo = pickle.load(f)

def calcular_percentual(imagem, cor_baixo, cor_alto):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, cor_baixo, cor_alto)
    percentual = (np.sum(mascara > 0) / mascara.size) * 100
    return percentual

def desenhar_grafico(valores):
    fig, ax = plt.subplots(figsize=(3, 4))
    canvas = FigureCanvas(fig)

    ax.barh(['Azul', 'Dourado', 'Prateado'], valores, color=['blue', 'gold', 'gray'])
    ax.set_xlim(0, 100)
    ax.set_title('Percentuais de Cor')
    ax.set_xlabel('%')

    plt.tight_layout()
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return buf

# Inicializa câmera
cap = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not cap.isOpened():
    print("Erro ao acessar a câmera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o quadro da câmera.")
        break

    # Define uma área de interesse (ROI centralizada)
    h, w = frame.shape[:2]
    roi = frame[h//4:h*3//4, w//4:w*3//4]

    # Calcula percentuais
    azul = calcular_percentual(roi, np.array([90, 50, 50]), np.array([140, 255, 255]))
    dourado = calcular_percentual(roi, np.array([20, 100, 100]), np.array([40, 255, 255]))
    prateado = calcular_percentual(roi, np.array([0, 0, 150]), np.array([180, 50, 255]))

    entrada = np.array([[azul, dourado, prateado]])
    predicao = modelo.predict(entrada)[0]

    # Desenha retângulo da ROI
    cv2.rectangle(frame, (w//4, h//4), (w*3//4, h*3//4), (255, 255, 0), 2)

    if predicao == 0:
        texto = 'PRATEADO'
        cor = (200, 200, 200)  # Cinza
    elif predicao == 1:
        texto = 'DOURADO'
        cor = (0, 215, 255)    # Dourado
    else:
        texto = 'QUEIMADO'
        cor = (0, 0, 255)      # Vermelho

    # Adiciona o texto no vídeo
    cv2.putText(frame, f'{texto}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)

    # Gera o gráfico
    grafico = desenhar_grafico([azul, dourado, prateado])
    grafico = cv2.resize(grafico, (300, frame.shape[0]))

    # Junta câmera + gráfico
    combinado = np.hstack((frame, grafico))

    cv2.imshow('Detector de Cavacos - Neuro Torno', combinado)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
