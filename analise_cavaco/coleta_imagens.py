import cv2
import os

# Pasta onde serão salvas as imagens
pasta_destino = 'cavacos/queimado'  # ou 'cavacos/normal', mudar conforme coleta

# Cria pasta se não existir
os.makedirs(pasta_destino, exist_ok=True)

# Inicializa câmera
cap = cv2.VideoCapture(0)
contador = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow('Captura de Cavacos', frame)

    # Pressionar 's' para salvar imagem
    if cv2.waitKey(1) & 0xFF == ord('s'):
        nome_arquivo = os.path.join(pasta_destino, f'cavaco_{contador}.jpg')
        cv2.imwrite(nome_arquivo, frame)
        print(f'[INFO] Imagem salva em: {nome_arquivo}')
        contador += 1

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
