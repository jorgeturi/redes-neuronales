import pandas as pd
import matplotlib.pyplot as plt

# Cargar el archivo CSV
archivo_csv = 'potencias2filtrado.csv'  # Nombre de tu archivo CSV
df = pd.read_csv(archivo_csv)

# Verificar si la columna 'activa' existe en el CSV
if 'activa' in df.columns:
    # Obtener los datos de la columna 'activa'
    datos = df['activa'].values
    total_datos = len(datos)
    
    # Definir los índices específicos para cada conjunto
    train_end = 71500    # Final del conjunto de entrenamiento
    val_end = 89600      # Final del conjunto de validación

    # Dividir los datos en tres partes
    X_train = datos[:train_end]        # 0 a 71500 (Entrenamiento)
    X_val = datos[train_end:val_end]   # 71500 a 89600 (Validación)
    X_test = datos[val_end:]           # 89600 en adelante (Prueba)

    # Calcular el porcentaje de cada conjunto
    train_percentage = (len(X_train) / total_datos) * 100
    val_percentage = (len(X_val) / total_datos) * 100
    test_percentage = (len(X_test) / total_datos) * 100

    # Imprimir la cantidad de datos en cada conjunto y su porcentaje
    print(f"Cantidad de datos en Entrenamiento: {len(X_train)} ({train_percentage:.2f}%)")
    print(f"Cantidad de datos en Validación: {len(X_val)} ({val_percentage:.2f}%)")
    print(f"Cantidad de datos en Prueba: {len(X_test)} ({test_percentage:.2f}%)")

    # Crear el gráfico
    plt.figure(figsize=(10, 6))

    # Graficar los tres subconjuntos con colores diferentes
    plt.plot(range(len(X_train)), X_train, label='Entrenamiento', color='blue')
    plt.plot(range(len(X_train), len(X_train) + len(X_val)), X_val, label='Validación', color='orange')
    plt.plot(range(len(X_train) + len(X_val), len(datos)), X_test, label='Prueba', color='green')

    # Agregar título y etiquetas
    plt.title('División de datos en entrenamiento, validación y prueba')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.legend()

    # Guardar la imagen como archivo PNG

    # Mostrar el gráfico
    plt.show()

else:
    print("La columna 'activa' no existe en el archivo CSV.")
