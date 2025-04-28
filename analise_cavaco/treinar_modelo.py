import numpy as np
from sklearn.svm import SVC
import pickle

# Carrega os dados
X = np.load('dados_X.npy')
y = np.load('dados_y.npy')

# Cria e treina o modelo
modelo = SVC(kernel='linear')
modelo.fit(X, y)

# Salva o modelo
with open('modelo_svm.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print('[INFO] Modelo treinado e salvo como modelo_svm.pkl.')
