import logging
from tools.logger_config import setup_logger
import tensorflow as tf
import numpy as np

import os

if __name__ == "__main__":
    # codigo principal
    setup_logger()    
    from tools.red_principal import *
    from tools.bd import leer_data_crear_df, escribir_bd

    modelo = cargar_modelo("/home/rnn/dev/modelo_trained.h5")
    if modelo is not None:  #si consegui el modelo
        modelo.summary()  
        scal = cargar_escaladores("/home/rnn/dev/scalers.pkl")
        if scal is not None:
            print("scalers:", scal)
            df = leer_data_crear_df()
            df = codificar_tiempo(df)
            print(df)
            X= crear_ventana_dataset(df,4)
            print(X)
            X_n = escalar_entrada(X,scal)


            logging.info("inicio prediccion")
            predicciones = modelo.predict(X_n, batch_size=1)
            print("fin prediccion")
            predicciones = scal['salidas'].inverse_transform(predicciones)
            print(predicciones)
            escribir_bd(predicciones)


