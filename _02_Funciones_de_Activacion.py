import numpy as np

# Funci�n de activaci�n sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Datos de entrada
input_data = np.array([-1, 0, 1])

# Calcular la salida utilizando la funci�n de activaci�n
output = sigmoid(input_data)
print("Salida de la funci�n de activaci�n sigmoide:", output)
