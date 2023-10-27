import numpy as np

# Definir las entradas y salidas de ejemplo
entradas = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
salidas = np.array([1, 1, 0])

# Inicializar los pesos sinápticos de manera aleatoria
pesos_sinapticos = np.random.rand(3)

# Parámetro de aprendizaje
tasa_aprendizaje = 0.1

# Número de épocas de entrenamiento
num_epocas = 1000

# Entrenamiento utilizando la regla de Hebb
for _ in range(num_epocas):
    for i in range(len(entradas)):
        entrada = entradas[i]
        salida_deseada = salidas[i]

        # Calcular la salida estimada
        salida_estimada = np.dot(entrada, pesos_sinapticos)

        # Actualizar los pesos utilizando la regla de Hebb
        delta_pesos = tasa_aprendizaje * (salida_deseada - salida_estimada) * entrada
        pesos_sinapticos += delta_pesos

# Imprimir los pesos sinápticos finales
print("Pesos Sinápticos Finales:", pesos_sinapticos)
