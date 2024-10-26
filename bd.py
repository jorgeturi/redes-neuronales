import numpy as np
import time
import datetime
from datetime import timedelta
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from pqsql import fetch_one_column_last_n_rows
from postgresql import insert_dataframe
import arguments_write as argsw
import arguments as args



import pandas as pd
from datetime import datetime

def leer_data_crear_df(muestras=4, maxdataP=1000):
    # Leer datos de la base de datos
    dataTime = fetch_one_column_last_n_rows(args.PSQL_ENGINE_URL, args.PSQL_FI_TABLE, 'time', muestras)
    P_in = fetch_one_column_last_n_rows(args.PSQL_ENGINE_URL, args.PSQL_FI_TABLE, 'power_w', muestras)

    # Invertir los vectores para que queden del dato más viejo al más nuevo
    dataTime = dataTime[::-1]
    P_in = P_in[::-1]

    # Normalizar los datos de potencia
    P = [a / maxdataP for a in P_in]

    # Obtener las fechas en formato "yyyy-mm-dd hh:mm"
    formatted_dates = [dt.strftime("%Y-%m-%d %H:%M") for dt in dataTime]

    # Crear el DataFrame
    df = pd.DataFrame({
        'timestamp': formatted_dates,
        'activa': P
    })
    
    return df





def escribir_bd(predicciones):
    dataTime = fetch_one_column_last_n_rows(args.PSQL_ENGINE_URL, args.PSQL_FI_TABLE, 'time', 4)
    dataTime = dataTime[::-1]


    tiempofuturo = dataTime[-1] + timedelta(minutes=10)  # Última fecha + 10 minutos
    print(tiempofuturo)


    timestamp_str = tiempofuturo.strftime('%Y-%m-%d %H:%M')  
    print(timestamp_str)

    # Rutina de escritura en la base de datos
    argsw.dataframe['time'] = timestamp_str
    argsw.dataframe['next_power_watt'] = float(predicciones[0, 0])

         # Cálculo del error de predicción de la muestra anterior
        #Pred_anterior = fetch_one_column_last_n_rows(args2.PSQL_ENGINE_URL, args2.PSQL_TABLE, 'next_power_watt', 1)
        #errorpred = Pred_anterior[0] - P_in[-1]  # Error de predicción respecto de la muestra anterior
        #print(errorpred)

    print(argsw.dataframe)

        #args2.dataframe['error'] = -1
    insert_dataframe(argsw.PSQL_ENGINE_URL, argsw.PSQL_TABLE, argsw.dataframe, False)
