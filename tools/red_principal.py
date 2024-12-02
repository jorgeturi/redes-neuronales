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




def crear_ventana(dataset, ventana_entrada, ventana_salida):
    logging.info("Creando ventanas.")

    # Extraer las características necesarias
    features = dataset[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']].values
    #features = dataset[['activa']].values


    # Calcular el número total de ventanas que se pueden crear
    total_muestras = len(dataset) - ventana_entrada - ventana_salida + 1

    # Ventanas de entrada
    X = np.array([features[i:i + ventana_entrada] for i in range(total_muestras)])

    # Ventanas de salida
    y = np.array([dataset['activa'].values[i + ventana_entrada:i + ventana_entrada + ventana_salida] for i in range(total_muestras)])

    logging.info("Ventanas creadas")

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

    dt2['dia_semana'] = dt2['timestamp'].dt.dayofweek
    dt2['sem_sen'] = np.sin(2 * np.pi * dt2['dia_semana'] / 7)
    dt2['sem_cos'] = np.cos(2 * np.pi * dt2['dia_semana'] / 7)


    dt2 = dt2[['activa', 'dia_sen', 'dia_cos', 'sem_sen', 'sem_cos']]
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




def cargar_datos_especificos(archivo_potencias='potencias.csv', archivo_corrientes='corrientes.csv', dias_semanales=None, horas=None):
    """
    Carga los datos desde los archivos CSV, filtra según días de la semana y horas, y registra el resultado en el log.

    Args:
    archivo_potencias (str): La ruta al archivo de potencias (csv).
    archivo_corrientes (str): La ruta al archivo de corrientes (csv).
    dias_semanales (list): Lista de días de la semana a cargar (0=domingo, 1=lunes, ..., 6=sábado). 
                           Si es None, no filtra por días de la semana.
    horas (list): Lista de horas (0-23) a cargar. Si es None, no filtra por horas.
    
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

        # Codificación del tiempo del día
        df_unido['tiempo_del_dia'] = df_unido['timestamp'].dt.hour + df_unido['timestamp'].dt.minute / 60.0
        df_unido['dia_sen'] = np.sin(2 * np.pi * df_unido['tiempo_del_dia'] / 24)
        df_unido['dia_cos'] = np.cos(2 * np.pi * df_unido['tiempo_del_dia'] / 24)

        # Codificación del día del año
        df_unido['dia_del_año'] = df_unido['timestamp'].dt.dayofyear
        df_unido['mes_sen'] = np.sin(2 * np.pi * df_unido['dia_del_año'] / 365)
        df_unido['mes_cos'] = np.cos(2 * np.pi * df_unido['dia_del_año'] / 365)

        # Normalización de la codificación cíclica a rango [0, 1]
        df_unido['dia_sen'] = (df_unido['dia_sen'] + 1) / 2
        df_unido['dia_cos'] = (df_unido['dia_cos'] + 1) / 2
        df_unido['mes_sen'] = (df_unido['mes_sen'] + 1) / 2
        df_unido['mes_cos'] = (df_unido['mes_cos'] + 1) / 2

        # Filtrar por días de la semana
        if dias_semanales is not None:
            df_unido = df_unido[df_unido['timestamp'].dt.dayofweek.isin(dias_semanales)]

        # Filtrar por horas
        if horas is not None:
            df_unido = df_unido[df_unido['timestamp'].dt.hour.isin(horas)]

        # Seleccionar y reorganizar las columnas en el formato deseado
        final_df = df_unido[['activa', 'dia_sen', 'dia_cos', 'mes_sen', 'mes_cos']]
        #final_df = df_unido[['activa']]


        logging.info(f"Datos cargados y filtrados. Total de registros: {len(final_df)}")

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







from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, BatchNormalization, Dense, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal


def entrenar_modelo(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado.h5'):
    # Asegurar reproducibilidad
    np.random.seed(47)
    tf.random.set_seed(47)
    initializer = GlorotUniform(seed=47)

    # Define los intervalos y los valores de learning rate
    boundaries = [1, 2, 20, 80, 140, 250]  # Los límites de los intervalos (épocas en este caso)
    values = [0.005, 0.002, 0.001, 0.0001, 0.00001, 0.00005, 0.000001]  # Learning rates correspondientes a los intervalos

    # Crea el scheduler de learning rate
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )
    # Crear el modelo LSTM
    model = Sequential()

    model.add(LSTM(256, return_sequences=True, input_shape=(Xtrain.shape[1], Xtrain.shape[2]), kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    model.add(BatchNormalization())

    #model.add(Dense(128, activation='relu', kernel_initializer=initializer))
    #model.add(Dropout(0.2)) 
    model.add(LSTM(128, return_sequences=True, kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False, kernel_initializer=initializer ) )
    #model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Dense(ytrain.shape[1], activation="sigmoid"))

    # Compilar el modelo con el optimizador personalizado
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1)
    model.compile(optimizer=optimizer, loss='huber_loss')

    # EarlyStopping para evitar sobreajuste
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # ModelCheckpoint para guardar el modelo durante el entrenamiento
    checkpoint = ModelCheckpoint(path_guardado, monitor='val_loss', save_best_only=True, verbose=1)

    try:
        # Entrenar el modelo con datos de validación, EarlyStopping y ModelCheckpoint
        model.fit(Xtrain, ytrain, epochs=250, verbose=1, batch_size=4,
                  validation_data=(Xval, yval), callbacks=[early_stopping, checkpoint])
    except MemoryError as e:
        print("Error de memoria: ", e)
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo al momento de fallo
    except Exception as e:
        print(f"Se produjo un error: {e}")
        print("Guardando el modelo hasta el último punto alcanzado...")
        model.save(path_guardado)  # Guarda el modelo si ocurre otro error

    return model



from keras.layers import Attention, Concatenate, LSTM, Bidirectional, Dropout, BatchNormalization, Dense, Input
from keras.models import Model
from keras.layers import MultiHeadAttention

def entrenar_modelo_con_atencion(Xtrain, ytrain, Xval, yval, path_guardado='modelo_entrenado.h5'):
    # Asegurar reproducibilidad
    np.random.seed(47)
    tf.random.set_seed(47)
    initializer = GlorotUniform(seed=47)

    # Definir el modelo
    inputs = Input(shape=(Xtrain.shape[1], Xtrain.shape[2]))  # (timesteps, features)
    
    # Capa LSTM Bidireccional
    lstm_out = Bidirectional(LSTM(64, return_sequences=True, kernel_initializer=initializer, kernel_regularizer=l2(0.01)))(inputs)
    lstm_out = Dropout(0.1)(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    # Aplicar el mecanismo de atención sobre la salida LSTM
    attention_out = MultiHeadAttention(num_heads=2, key_dim=10)(lstm_out, lstm_out)
    print(attention_out.shape)  # (None, 24, 128)
    
    # Aplicar GlobalAveragePooling1D para reducir la secuencia
    pooled_out = GlobalAveragePooling1D()(attention_out)  # Esto reducirá la salida a (None, 128)
    
    # Capa de salida
    output = Dense(ytrain.shape[1], activation="relu" , kernel_initializer=initializer)(pooled_out)

    # Crear el modelo
    model = Model(inputs=inputs, outputs=output)

    # Definir el optimizador y la compilación del modelo
    boundaries = [2, 3, 4, 10, 100, 250]
    values = [0.04, 0.005, 0.002, 0.0001, 0.00001, 0.00005, 0.000001]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries,
        values=values
    )
    optimizer = Adam(learning_rate=lr_schedule, clipnorm=1)
    model.compile(optimizer=optimizer, loss='mse')

    # Configuración de callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    checkpoint = ModelCheckpoint(path_guardado, monitor='val_loss', save_best_only=True, verbose=1)

    # Entrenamiento del modelo
    model.fit(Xtrain, ytrain, epochs=1, batch_size=1, validation_data=(Xval, yval), callbacks=[early_stopping, checkpoint])

    return model
