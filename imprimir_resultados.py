import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar el archivo CSV
archivo = 'resultados_predicciones_con_datos.csv'  # Reemplaza 'archivo.csv' con el nombre de tu archivo
df = pd.read_csv(archivo)

# Suponemos que el archivo tiene las columnas 'valor_real' y 'prediccion'
yval = df['valor_real'].tolist()  # Valores reales
prediccionesval = df['prediccion'].tolist()  # Predicciones

# Calcular el error relativo porcentual
error_relativo = [(abs(pred - real) / real) * 100 if real != 0 else 0 for pred, real in zip(prediccionesval, yval)]

# Graficar los valores reales, las predicciones y el error relativo
plt.figure(figsize=(10, 6))

# Graficar valores reales
plt.plot(yval[200:400], label='Valor Real',  linestyle='-',color='blue')

# Graficar predicciones
plt.plot(prediccionesval[201:400], label='Predicción', linestyle='-', color='red' )

# Graficar error relativo
#plt.plot(error_relativo[0:200], label='Error Relativo (%)', color='green', linestyle=':', marker='s')

# Títulos y etiquetas
plt.title('Valor Real vs Predicción con Error Relativo')
plt.xlabel('Índice de Observación')
plt.ylabel('Valor / Error Relativo (%)')
plt.legend()
plt.grid(True)

# Mostrar gráfico
plt.show()
