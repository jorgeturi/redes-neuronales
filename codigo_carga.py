import logging
from tools.logger_config import setup_logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os

if __name__ == "__main__":
    # codigo principal
    setup_logger()    
    from tools.red_principal import *
    #from tools.bd import leer_data_crear_df, escribir_bd

    modelo = cargar_modelo("modelo.keras")
    if modelo is not None:  #si consegui el modelo
        modelo.summary()  
        scal = cargar_escaladores("scalers.pkl")
        if scal is not None:
            print("scalers:", scal)
            #df = leer_data_crear_df()
            df = cargar_datos()
            #df = codificar_tiempo(df)
            #X= crear_ventana_dataset(df,4)
            X, y = crear_ventana(df[0:8000], 4*24,4*6)
            
            inicio_train = 0
            fin_train = 3000
            inicio_val = fin_train+1
            fin_val = fin_train+1+1000
            inicio_test = fin_val+1
            fin_test = fin_val+1+1000
            # conjunto de validación
            Xval = X[inicio_val:fin_val]
            yval = y[inicio_val:fin_val]
            #conjunto de entrenamiento
            Xtrain = X[inicio_train:fin_train]
            ytrain = y[inicio_train:fin_train]

            Xtest = X[inicio_test:fin_test]
            ytest = X[inicio_test:fin_test]

            #X_n = escalar_entrada(X,scal)


            logging.info("inicio prediccion")
            prediccionesval = modelo.predict(Xval, batch_size=1)
            prediccionestest = modelo.predict(Xtest, batch_size=1)
            print("fin prediccion")
            prediccionesval[0]
            print("comparo con los valores reales")
            yval[0]
            import pandas as pd

            # Crear listas para almacenar los resultados
            valores_reales = []
            predicciones = []
            errores = []
            resultados = []
            errores_totales = []

            for valor in range(len(yval)):
                y_real = yval[valor]
                prediccion = prediccionesval[valor]
                
                # Calcular errores y almacenarlos
                errores = []
                for i in range(len(y_real)):
                    error = prediccion[i] - y_real[i]
                    errores.append(error)
                    resultados.append({
                        'valor': valor,
                        'yval': y_real[i],
                        'prediccion': prediccion[i],
                        'error': error
                    })

                # Calcular el error promedio para el conjunto actual
                error_promedio = np.mean(np.abs(errores))
                errores_totales.append(error_promedio)

            # Convertir a DataFrame
            df_resultados = pd.DataFrame(resultados)

            # Calcular el error promedio total
            error_promedio_total = np.mean(np.abs(df_resultados['error']))

            # Graficar error promedio por conjunto
            plt.figure(figsize=(10, 5))
            plt.scatter(range(len(errores_totales)), errores_totales, color='skyblue')
            plt.title('Error Promedio por Conjunto')
            plt.xlabel('Conjunto de Valores')
            plt.ylabel('Error Promedio')
            plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            plt.grid()

            # Agregar el error promedio total al gráfico
            plt.annotate(f'Error Promedio Total: {error_promedio_total:.2f}', xy=(0, error_promedio_total), 
                        xytext=(1, error_promedio_total + 0.5),
                        arrowprops=dict(facecolor='black', shrink=0.05))

            plt.show()

            # Guardar resultados en CSV
            df_resultados.to_csv('predicciones_detalle.csv', index=False)

            print("Archivo 'predicciones_detalle.csv' creado con éxito.")