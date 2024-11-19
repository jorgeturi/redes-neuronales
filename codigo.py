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
            dias = [1,2,3,4,5]  # Lunes, Miércoles, Viernes
            horas = [0,1,2,3,4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23]  # De 9 a 12 horas
            df = cargar_datos_especificos('potencias.csv', 'corrientes.csv', dias_semanales=dias, horas=horas)
            #df = codificar_tiempo(df)
            #X= crear_ventana_dataset(df,4)
            print(df.shape)
            X, y = crear_ventana(df[23000:80000], 4*24,4*12)
            
            inicio_train = 0
            fin_train = 15000
            inicio_val = fin_train+1
            fin_val = fin_train+1+5000
            # conjunto de validación
            Xval = X[inicio_val:fin_val]
            yval = y[inicio_val:fin_val]
            #conjunto de entrenamiento
            Xtrain = X[inicio_train:fin_train]
            ytrain = y[inicio_train:fin_train]

            from sklearn.preprocessing import MinMaxScaler
            scaleractiva = MinMaxScaler(feature_range=(-1, 1))
            Xtrain_n = Xtrain.copy()
            Xtrain_n[:, :, 0] = scaleractiva.fit_transform(Xtrain[:, :, 0])
            Xval_n = Xval.copy()
            Xval_n[:, :, 0] = scaleractiva.transform(Xval[:, :, 0])
            print(Xval_n)

            salidas = MinMaxScaler(feature_range=(-1, 1))
            ytrain_n = ytrain.copy()
            ytrain_n = salidas.fit_transform(ytrain)
            yval_n = salidas.transform(yval)
            print(yval_n)

            
            import pickle
            scalers = {'scaleractiva': scaleractiva, 'salidas': salidas}

            with open('scalers.pkl', 'wb') as f:
                pickle.dump(scalers, f)


            #X_n = escalar_entrada(X,scal)
            from sklearn.utils import shuffle

            Xtrain_n, ytrain_n = shuffle(Xtrain_n, ytrain_n, random_state=0)
            #Xval, yval = shuffle(Xval, yval, random_state=0)


            
            logging.info("inicio prediccion")
            #predicciones = modelo.predict(X_n, batch_size=1)
            modell = entrenar_modelo(Xtrain_n,ytrain_n, Xval_n, yval_n)
            modell.save('modelo.keras')
            print("fin prediccion")
            #predicciones = modelo.predict(X_n, batch_size=1)

            #predicciones = scal['salidas'].inverse_transform(predicciones)
            #print(predicciones)
            #escribir_bd(predicciones)


