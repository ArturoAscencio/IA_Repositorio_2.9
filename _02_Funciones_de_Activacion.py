import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Datos de entrada
input_data = np.array([-1, 0, 1])

# Calcular la salida utilizando la función de activación
output = sigmoid(input_data)
print("Salida de la función de activación sigmoide:", output)
