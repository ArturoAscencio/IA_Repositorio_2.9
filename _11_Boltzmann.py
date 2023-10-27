import numpy as np

# Funci�n de activaci�n sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Datos de entrada (aleatorios)
data = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [1, 0, 0, 1]])

# Hiperpar�metros
num_visible = data.shape[1]
num_hidden = 2
learning_rate = 0.1
epochs = 1000

# Inicializaci�n de pesos y sesgos
weights = np.random.rand(num_visible, num_hidden)
visible_bias = np.zeros(num_visible)
hidden_bias = np.zeros(num_hidden)

# Entrenamiento de la RBM
for epoch in range(epochs):
    for v in data:
        # Paso positivo: Probabilidades de activaci�n de las unidades ocultas
        prob_hidden_given_visible = sigmoid(np.dot(v, weights) + hidden_bias)
        # Muestreo de unidades ocultas
        hidden_states = (prob_hidden_given_visible > np.random.rand(num_hidden)).astype(int)

        # Paso negativo: Reconstrucci�n de las unidades visibles
        prob_visible_given_hidden = sigmoid(np.dot(hidden_states, weights.T) + visible_bias)
        # Muestreo de unidades visibles
        visible_states = (prob_visible_given_hidden > np.random.rand(num_visible)).astype(int)

        # Actualizaci�n de los pesos y sesgos
        delta_weights = np.outer(v, prob_hidden_given_visible) - np.outer(visible_states, prob_visible_given_hidden)
        weights += learning_rate * delta_weights
        hidden_bias += learning_rate * (prob_hidden_given_visible - prob_visible_given_hidden)
        visible_bias += learning_rate * (v - visible_states)

# Imprimir los pesos finales
print("Pesos finales:")
print(weights)
