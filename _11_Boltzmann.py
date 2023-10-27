import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Datos de entrada (aleatorios)
data = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [1, 0, 0, 1]])

# Hiperparámetros
num_visible = data.shape[1]
num_hidden = 2
learning_rate = 0.1
epochs = 1000

# Inicialización de pesos y sesgos
weights = np.random.rand(num_visible, num_hidden)
visible_bias = np.zeros(num_visible)
hidden_bias = np.zeros(num_hidden)

# Entrenamiento de la RBM
for epoch in range(epochs):
    for v in data:
        # Paso positivo: Probabilidades de activación de las unidades ocultas
        prob_hidden_given_visible = sigmoid(np.dot(v, weights) + hidden_bias)
        # Muestreo de unidades ocultas
        hidden_states = (prob_hidden_given_visible > np.random.rand(num_hidden)).astype(int)

        # Paso negativo: Reconstrucción de las unidades visibles
        prob_visible_given_hidden = sigmoid(np.dot(hidden_states, weights.T) + visible_bias)
        # Muestreo de unidades visibles
        visible_states = (prob_visible_given_hidden > np.random.rand(num_visible)).astype(int)

        # Actualización de los pesos y sesgos
        delta_weights = np.outer(v, prob_hidden_given_visible) - np.outer(visible_states, prob_visible_given_hidden)
        weights += learning_rate * delta_weights
        hidden_bias += learning_rate * (prob_hidden_given_visible - prob_visible_given_hidden)
        visible_bias += learning_rate * (v - visible_states)

# Imprimir los pesos finales
print("Pesos finales:")
print(weights)
