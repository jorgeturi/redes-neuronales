import numpy as np
import time
import datetime
from datetime import timedelta
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from pqsql import fetch_one_column_last_n_rows, get_column_labels
import arguments as args
from postgresql import insert_dataframe
import arguments_write as args2

# Parámetros
muestras = 4
MaxdataP = 1000  # Define el valor máximo de power_w (ajústalo según sea necesario)

# Inicialización
i = 1
Prediccion = 0

print(i)

    # Leer datos de la base de datos
dataTime = fetch_one_column_last_n_rows(args.PSQL_ENGINE_URL, args.PSQL_FI_TABLE, 'time', muestras)
P_in = fetch_one_column_last_n_rows(args.PSQL_ENGINE_URL, args.PSQL_FI_TABLE, 'power_w', muestras)

#P_in = P_in /1000 ###esta en el codigo ppal como potencia maxima que se mide para obtener la pot

    # Invertir los vectores para que queden del dato más viejo al más nuevo
dataTime = dataTime[::-1]
P_in = P_in[::-1]

P = []
for n in range(muestras):
        a = P_in[n]
        P.append(a / MaxdataP)

#Parray = np.array(P).reshape(1, muestras)

print(dataTime)
print(" \n")
print(P)
print(" \n")

from datetime import datetime, timezone, timedelta

# Obtener las fechas en formato "yyyy-mm-dd hh:mm"
formatted_dates = [dt.strftime("%Y-%m-%d %H:%M") for dt in dataTime]

# Imprimir las fechas formateadas
for date in formatted_dates:
    print(date)





test = get_column_labels()
print(test)

import pandas as pd

df = pd.DataFrame({
    'timestamp': formatted_dates,
    'activa': P
})
print(df)



import logging
from tools.logger_config import setup_logger
import tensorflow as tf
import numpy as np


from tools.red_principal import *

modelo = cargar_modelo("/home/rnn/dev/modelo_trained.h5")
if modelo is not None:  #si consegui el modelo
    modelo.summary()  
    scal = cargar_escaladores("/home/rnn/dev/scalers.pkl")
    if scal is not None:
        print("scalers:", scal)
        df = codificar_tiempo(df)
        print(df)
        #datos = cargar_datos("/home/rnn/dev/potencias.csv", "/home/rnn/dev/corrientes.csv")
        X= crear_ventana_dataset(df,4)
        print(X)
        X_n = escalar_entrada(X,scal)

        predicciones = modelo.predict(X_n, batch_size=1)
        predicciones = scal['salidas'].inverse_transform(predicciones)

        print("fin prediccion")
        print(predicciones[0, 0])
        print(type(predicciones[0, 0]))


        tiempofuturo = dataTime[-1] + timedelta(minutes=10)  # Última fecha + 10 minutos
        print(tiempofuturo)


        timestamp_str = tiempofuturo.strftime('%Y-%m-%d %H:%M')  
        print(timestamp_str)

        # Rutina de escritura en la base de datos
        args2.dataframe['time'] = timestamp_str
        args2.dataframe['next_power_watt'] = float(predicciones[0, 0])

         # Cálculo del error de predicción de la muestra anterior
        #Pred_anterior = fetch_one_column_last_n_rows(args2.PSQL_ENGINE_URL, args2.PSQL_TABLE, 'next_power_watt', 1)
        #errorpred = Pred_anterior[0] - P_in[-1]  # Error de predicción respecto de la muestra anterior
        #print(errorpred)

        print(args2.dataframe)

        #args2.dataframe['error'] = -1
        insert_dataframe(args2.PSQL_ENGINE_URL, args2.PSQL_TABLE, args2.dataframe, False)


