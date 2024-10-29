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

    modelo = cargar_modelo("modelo_trained.keras")
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
            fin_train = 2000
            inicio_val = fin_train+1
            fin_val = fin_train+1+500
            # conjunto de validación
            Xval = X[inicio_val:fin_val]
            yval = y[inicio_val:fin_val]
            #conjunto de entrenamiento
            Xtrain = X[inicio_train:fin_train]
            ytrain = y[inicio_train:fin_train]

            #X_n = escalar_entrada(X,scal)

            
            logging.info("inicio prediccion")
            #predicciones = modelo.predict(X_n, batch_size=1)
            modell = entrenar_modelo(Xtrain,ytrain, Xval, yval)
            modell.save('modelo.keras')
            print("fin prediccion")
            #predicciones = modelo.predict(X_n, batch_size=1)

            #predicciones = scal['salidas'].inverse_transform(predicciones)
            #print(predicciones)
            #escribir_bd(predicciones)


