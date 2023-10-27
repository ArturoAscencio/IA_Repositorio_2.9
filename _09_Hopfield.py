import numpy as np

class HopfieldNetwork:
    def __init__(self, pattern_size):
        self.pattern_size = pattern_size
        self.weights = np.zeros((pattern_size, pattern_size))

    def train(self, patterns):
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
            np.fill_diagonal(self.weights, 0)

    def update(self, pattern):
        pattern = np.array(pattern)
        for _ in range(self.pattern_size):
            activations = np.dot(self.weights, pattern)
            pattern = np.where(activations >= 0, 1, -1)
        return pattern

if __name__ == "__main__":
    # Crear una red de Hopfield para patrones de 9 elementos
    pattern_size = 9
    hopfield_net = HopfieldNetwork(pattern_size)

    # Definir patrones para almacenar
    patterns = [
        [1, 1, 1, -1, -1, -1, -1, 1, 1],
        [1, -1, -1, -1, 1, -1, -1, -1, 1],
        [-1, 1, 1, -1, 1, -1, 1, 1, -1]
    ]

    # Entrenar la red con los patrones
    hopfield_net.train(patterns)

    # Patrón de prueba para recuperación
    test_pattern = [1, -1, 1, -1, 1, -1, -1, -1, -1]

    # Actualizar el patrón de prueba
    updated_pattern = hopfield_net.update(test_pattern)

    # Imprimir el patrón original y el patrón recuperado
    print("Patrón original:")
    print(np.array(test_pattern).reshape(3, 3))
    print("Patrón recuperado:")
    print(updated_pattern.reshape(3, 3))
