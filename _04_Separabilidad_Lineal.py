pip install numpy

import numpy as np

# Definir un conjunto de datos con separabilidad lineal
# Usaremos dos características (X1 y X2) y dos clases (0 y 1)
X = np.array([[1, 2],
              [2, 3],
              [3, 2],
              [6, 5],
              [7, 7],
              [8, 6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Inicializar los pesos y el sesgo
np.random.seed(0)
w = np.random.rand(2)  # Pesos
b = np.random.rand(1)  # Sesgo

# Tasa de aprendizaje
learning_rate = 0.1

# Entrenar el Perceptrón
epochs = 10
for _ in range(epochs):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        a = 1 if z > 0 else 0
        error = y[i] - a
        w += learning_rate * error * X[i]
        b += learning_rate * error

# Comprobar la precisión del Perceptrón
correct = 0
for i in range(len(X)):
    z = np.dot(X[i], w) + b
    a = 1 if z > 0 else 0
    if a == y[i]:
        correct += 1

accuracy = correct / len(X)
print("Precisión del Perceptrón:", accuracy)

# Imprimir los pesos y el sesgo finales
print("Pesos finales:", w)
print("Sesgo final:", b)
