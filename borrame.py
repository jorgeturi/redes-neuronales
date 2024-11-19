import numpy as np
import matplotlib.pyplot as plt

# Días de la semana (lunes = 0, martes = 1, ..., domingo = 6)
dias = np.array([0, 1, 2, 3, 4, 5, 6])
nombres_dias = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Codificación trigonométrica
angulo = (2 * np.pi / 7) * dias
x = np.cos(angulo)  # Componente coseno
y = np.sin(angulo)  # Componente seno

# Crear gráfico cartesiano
plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='blue')  # Graficar los puntos

# Etiquetas para los días de la semana y sus coordenadas
for i, dia in enumerate(dias):
    # Etiquetar el nombre del día
    plt.text(x[i] + 0.05, y[i] + 0.05, nombres_dias[i], ha='center', va='center', fontsize=12)
    # Etiquetar las coordenadas (x, y) de cada punto
    plt.text(x[i] + 0.05, y[i] - 0.05, f'({x[i]:.2f}, {y[i]:.2f})', ha='center', va='center', fontsize=10, color='red')

# Añadir el círculo unitario
circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--')
plt.gca().add_artist(circle)

# Establecer límites y aspecto igual para mantener la proporción circular
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.gca().set_aspect('equal', adjustable='box')

# Título y mostrar el gráfico
plt.title("Circular Encoding of the Days of the Week")
plt.grid(True)
plt.show()
