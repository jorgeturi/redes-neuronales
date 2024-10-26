import logging
import numpy as np
from tensorflow.keras.models import load_model  # Cargar modelo
import joblib #cargar escaladores
import pandas as pd # cargar datos 

def funcion():
    logging.info("Este es un mensaje desde la función.")
    logging.debug("Este es un mensaje de depuración en la función.")
    logging.warning("Este es un mensaje de advertencia en la función.")
    logging.error("Este es un mensaje de error en la función.")

def cargar_modelo(ruta_al_modelo="default.txt"):
    """
    Carga un modelo desde un archivo y registra el resultado en el log.

    Args:
    ruta_al_modelo (str): La ruta al archivo del modelo.

    Returns:
    model: El modelo cargado si la carga fue exitosa, None sino.
    """
    try:
        # Intentar cargar el modelo
        model = load_model(ruta_al_modelo)
        logging.info(f'Modelo cargado exitosamente desde {ruta_al_modelo}.')
        return model
    except Exception as e:
        logging.error(f'Error al cargar el modelo desde {ruta_al_modelo}: {e}')
        return None

def cargar_escaladores(ruta_a_escaladores="rutafake.txt"):
    """
    Carga escaladores desde un archivo y registra el resultado en el log.

    Args:
    ruta_a_escaladores (str): La ruta a escaladores del modelo.

    Returns:
    escaladores: escaladores cargados si la carga fue exitosa, None sino.
    """
    try:
        # Intentar cargar el modelo
        scalers = joblib.load(ruta_a_escaladores)

        logging.info(f'Escaladores cargados exitosamente desde {ruta_a_escaladores}.')
        return scalers
    except Exception as e:
        logging.error(f'Error al cargar escaladores desde {ruta_a_escaladores}: {e}')
        return None




def crear_ventana(dataset, ventana):
    X = []
    y = []
    logging.info("Creando ventanas.")

    # Crear ventanas y calcular el valor objetivo
    for i in range(len(dataset) - ventana):
        # Crear la ventana
        window = dataset.iloc[i:i+ventana].copy()

        # Características de la ventana
        window_features = window[['activa','dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']].values
        
        # El valor objetivo es el valor de 'activa' en el tiempo t+1
        target_value = dataset.iloc[i+ventana]['activa']

        X.append(window_features)
        y.append(target_value)

    # Convertir las listas a arrays de NumPy para facilitar su uso
    X = np.array(X)
    y = np.array(y)
    logging.info("ventanas creadas")

    return X, y

import numpy as np
import logging

def crear_ventana_dataset(dataset, ventana):
    X = []
    logging.info("Creando ventanas.")

    # veo si es posible
    if len(dataset) < ventana:
        logging.warning("El tamaño del dataset es menor que la ventana. No se pueden crear ventanas.")
        return np.array(X)  # Returno vacio

    # Crear ventanas
    for i in range(len(dataset) - ventana + 1):  
        # Crear la ventana
        window = dataset.iloc[i:i + ventana].copy()

        # Características de la ventana
        window_features = window[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']].values
        
        X.append(window_features)

    # Convertir las listas a arrays de NumPy para facilitar su uso
    X = np.array(X)
    logging.info("Ventanas creadas")

    return X




def codificar_tiempo(dt):
    dt2 = dt.copy()  

    # Ensure the 'timestamp' column is in datetime format
    dt2['timestamp'] = pd.to_datetime(dt2['timestamp'], errors='coerce')

    # Check if any timestamps failed to convert
    if dt2['timestamp'].isnull().any():
        raise ValueError("Some timestamps could not be converted to datetime format.")

    # Separar la columna de timestamp en año, mes, día, hora, minuto
    dt2['año'] = dt2['timestamp'].dt.year
    dt2['mes'] = dt2['timestamp'].dt.month
    dt2['dia'] = dt2['timestamp'].dt.day
    dt2['hora'] = dt2['timestamp'].dt.hour
    dt2['minuto'] = dt2['timestamp'].dt.minute

    # Codificación del tiempo del día
    dt2['tiempo_del_dia'] = dt2['hora'] + dt2['minuto'] / 60.0
    dt2['dia_sen'] = np.sin(2 * np.pi * dt2['tiempo_del_dia'] / 24)
    dt2['dia_cos'] = np.cos(2 * np.pi * dt2['tiempo_del_dia'] / 24)

    # Codificación del día del año
    dt2['dia_del_año'] = dt2['timestamp'].dt.dayofyear
    dt2['mes_sen'] = np.sin(2 * np.pi * dt2['dia_del_año'] / 365)
    dt2['mes_cos'] = np.cos(2 * np.pi * dt2['dia_del_año'] / 365)

    dt2 = dt2[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']]
    logging.info("tiempo codificado.")

    return dt2



def cargar_datos(archivo_potencias='potencias.csv', archivo_corrientes='corrientes.csv'):
    """
    Carga los datos desde los archivos y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    archivo_corrientes (str): La ruta al archivo de corrientes (csv)

    Returns:
    final_df: dataframe con los datos si la carga fue exitosa, None sino.
    """
    try:
        # Leer encabezados para verificar que los archivos se pueden abrir
        encabezados_corrientes = pd.read_csv(archivo_corrientes, nrows=0).columns
        encabezados_potencias = pd.read_csv(archivo_potencias, nrows=0).columns
        
        fila_inicio = 1
        numero_filas = 120000  # Número de filas que deseas cargar
        
        # Leer los archivos CSV
        corrientes = pd.read_csv(archivo_corrientes, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_corrientes)
        potencias = pd.read_csv(archivo_potencias, skiprows=fila_inicio, nrows=numero_filas, header=None, names=encabezados_potencias)

        # Convertir la columna 'timestamp' a datetime
        corrientes['timestamp'] = pd.to_datetime(corrientes['timestamp'])
        potencias['timestamp'] = pd.to_datetime(potencias['timestamp'])

        # Unir los dataframes en base al ID y timestamp
        df_unido = pd.merge(corrientes, potencias, on=['id', 'timestamp'])

        # Separar la columna de timestamp en año, mes, día, hora, minuto
        df_unido['año'] = df_unido['timestamp'].dt.year
        df_unido['mes'] = df_unido['timestamp'].dt.month
        df_unido['dia'] = df_unido['timestamp'].dt.day
        df_unido['hora'] = df_unido['timestamp'].dt.hour
        df_unido['minuto'] = df_unido['timestamp'].dt.minute

        # Codificación del tiempo del día
        df_unido['tiempo_del_dia'] = df_unido['hora'] + df_unido['minuto'] / 60.0
        df_unido['dia_sen'] = np.sin(2 * np.pi * df_unido['tiempo_del_dia'] / 24)
        df_unido['dia_cos'] = np.cos(2 * np.pi * df_unido['tiempo_del_dia'] / 24)

        # Codificación del día del año
        df_unido['dia_del_año'] = df_unido['timestamp'].dt.dayofyear
        df_unido['mes_sen'] = np.sin(2 * np.pi * df_unido['dia_del_año'] / 365)
        df_unido['mes_cos'] = np.cos(2 * np.pi * df_unido['dia_del_año'] / 365)

        # Seleccionar y reorganizar las columnas en el formato deseado
        final_df = df_unido[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']]
        logging.info("Datos cargados.")

        if np.any(np.isnan(final_df)):
                print("Hay valores NaN en los datos.")
        if np.any(np.isinf(final_df)):
                print("Hay valores infinitos en los datos.")
        return final_df
    
    except Exception as e:
        logging.error(f'Error al cargar los datos: {e}')
        return None


def escalar_datos(Xtrain, ytrain, scalers):
    Xtrain_n = Xtrain.copy()

    Xtrain_n[:, :, 0] = scalers['scaleractiva'].transform(Xtrain[:, :, 0])
    Xtrain_n[:, :, 1] = scalers['scalersenhora'].transform(Xtrain[:, :, 1]) 
    Xtrain_n[:, :, 2] = scalers['scalercoshora'].transform(Xtrain[:, :, 2])
    Xtrain_n[:, :, 3] = scalers['scalersendia'].transform(Xtrain[:, :, 3])
    Xtrain_n[:, :, 4] = scalers['scalercosdia'].transform(Xtrain[:, :, 4])

    ytrain_n = ytrain.copy()
    ytrain_n = ytrain_n.reshape(-1, 1)  
    ytrain_n = scalers['salidas'].transform(ytrain_n)  

    logging.info("datos escalados")
    return Xtrain_n, ytrain_n


def escalar_entrada(Xtrain, scalers):
    Xtrain_n = Xtrain.copy()

    Xtrain_n[:, :, 0] = scalers['scaleractiva'].transform(Xtrain[:, :, 0])
    Xtrain_n[:, :, 1] = scalers['scalersenhora'].transform(Xtrain[:, :, 1]) 
    Xtrain_n[:, :, 2] = scalers['scalercoshora'].transform(Xtrain[:, :, 2])
    Xtrain_n[:, :, 3] = scalers['scalersendia'].transform(Xtrain[:, :, 3])
    Xtrain_n[:, :, 4] = scalers['scalercosdia'].transform(Xtrain[:, :, 4])

    logging.info("entrada escalada")
    return Xtrain_n