import numpy as np

def actualizar_neurona(neurona, patrones):
    n = len(neurona)
    for i in range(n):
        suma = 0
        for j in range(n):
            suma += patrones[j] * neurona[j]
        if suma >= 0:
            neurona[i] = 1
        else:
            neurona[i] = -1
    return neurona

def almacenar_patrones(patrones):
    n = len(patrones[0])
    W = np.zeros((n, n))
    for p in patrones:
        W += np.outer(p, p)
    for i in range(n):
        W[i, i] = 0
    return W

def recuperar_patron(patrones, W, patron_incompleto):
    patron_recuperado = np.copy(patron_incompleto)
    for _ in range(5):  # Número de iteraciones de actualización
        patron_recuperado = actualizar_neurona(patron_recuperado, patron_recuperado)
    return patron_recuperado

# Ejemplo de uso
patrones = [np.array([1, -1, 1, -1]), np.array([1, 1, -1, -1]), np.array([-1, 1, 1, -1])]
W = almacenar_patrones(patrones)
patron_incompleto = np.array([1, -1, 0, 0])
patron_recuperado = recuperar_patron(patrones, W, patron_incompleto)

print("Patrón original:", patron_incompleto)
print("Patrón recuperado:", patron_recuperado)
