from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt

# Crear datos de ejemplo
data = np.random.rand(100, 2)

# Crear un SOM
som = MiniSom(10, 10, 2, sigma=1.0, learning_rate=0.5)

# Inicializar el SOM
som.random_weights_init(data)

# Entrenar el SOM
som.train_random(data, 100)

# Obtener las coordenadas de los mapas
win_map = som.win_map(data)

# Graficar el SOM
plt.figure(figsize=(6, 6))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.2)
plt.colorbar()

for position, data_points in win_map.items():
    data_points = np.array(data_points)
    plt.plot(position[0] + 0.5, position[1] + 0.5, 'o', markerfacecolor='None', markeredgecolor='red', markersize=12, markeredgewidth=2)
    
plt.title('SOM')
plt.show()
