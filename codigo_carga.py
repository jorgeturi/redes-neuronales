import logging
from tools.logger_config import setup_logger
import tensorflow as tf
import numpy as np

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

            # Recorrer cada conjunto de predicciones y valores reales
            for i in range(len(prediccionesval)):
              # Obtener el valor real y la predicción
                y_real = yval[i]
                prediccion = prediccionesval[i]
    
                # Calcular el error
                error = prediccion - y_real
    
                # Almacenar los resultados
                valores_reales.append(y_real.flatten())
                predicciones.append(prediccion.flatten())
                errores.append(error.flatten())

            #            Convertir listas a DataFrame
            resultados = pd.DataFrame({
                'yval': [list(y) for y in valores_reales],          # Valores reales
                'predicciones': [list(p) for p in predicciones],    # Predicciones
                'error': [list(e) for e in errores]                 # Errores
            })

            # Guardar el DataFrame en un archivo CSV
            resultados.to_csv('predicciones.csv', index=False)

            print("Archivo 'predicciones.csv' creado con éxito.")

