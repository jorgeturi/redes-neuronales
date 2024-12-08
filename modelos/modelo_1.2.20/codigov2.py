import os
import shutil
import logging
import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from tools.logger_config import setup_logger
from tools.red_principal import *  # Suponiendo que este es el script que quieres guardar
from sklearn.preprocessing import MinMaxScaler

def crear_carpeta_y_guardar(nombre_modelo):
    # Crear la carpeta con el nombre del modelo
    carpeta = f"modelos/{nombre_modelo}"
    os.makedirs(carpeta, exist_ok=True)

    # Copiar el script 'red_principal.py' a la carpeta
    ruta_script = 'tools/red_principal.py'  # Ruta del script
    destino_script = os.path.join(carpeta, 'red_principal.py')
    shutil.copy(ruta_script, destino_script)

    ruta_script = 'codigov2.py'  # Ruta del script
    destino_script = os.path.join(carpeta, 'codigov2.py')
    shutil.copy(ruta_script, destino_script)

    # Crear un archivo de resultados
    resultados_path = os.path.join(carpeta, 'resultados.txt')

    # Abrir el archivo de resultados para escribir
    with open(resultados_path, 'w') as f:
        f.write(f"Resultados de la predicción para el modelo {nombre_modelo}:\n")

    return carpeta, resultados_path

def guardar_modelo_y_resultados(carpeta, modelo, scalers):
    # Guardar el modelo
    modelo_path = os.path.join(carpeta, 'modelo')
    modelo.save(modelo_path)

    # Guardar los escaladores
    scalers_path = os.path.join(carpeta, 'scalers.pkl')
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)


    return modelo_path, scalers_path, resultados_path

import os
import numpy as np
import pandas as pd

def calcular_resultados(ytest, prediccionesTest, carpeta):
    # Crear listas para almacenar los resultados
    resultados = []
    errores_totales = []  # Para almacenar todos los errores

    for valor in range(len(ytest)):
        y_real = ytest[valor]
        prediccion = prediccionesTest[valor]
        
        # Calcular errores
        errores = [prediccion[i] - y_real[i] for i in range(len(y_real))]
        errores_totales.extend(errores)  # Guardamos todos los errores para análisis global

        # Calcular error promedio y desviación estándar
        error_promedio = np.mean(np.abs(errores))
        desviacion_estandar = np.std(errores)
        
        # Error relativo porcentual
        error_relativo_porcentual = [(abs(errores[i]) / abs(y_real[i])) * 100 if y_real[i] != 0 else 0 for i in range(len(y_real))]
        
        # Almacenar resultados
        for i in range(len(y_real)):
            resultados.append({
                'valor': valor,
                'prediccion': prediccion[i],
                'valor_real': y_real[i],
                'error': errores[i],
                'error_promedio': error_promedio,
                'desviacion_estandar': desviacion_estandar,
                'error_relativo_porcentual': error_relativo_porcentual[i]
            })

    # Convertir a DataFrame
    df_resultados = pd.DataFrame(resultados)

    # Calcular métricas globales
    error_promedio_global = np.mean(df_resultados['error_promedio'])
    desviacion_estandar_global = np.std(df_resultados['desviacion_estandar'])
    error_relativo_porcentual_promedio = np.mean(df_resultados['error_relativo_porcentual'])

    # Calcular las métricas de errores menores a 1, 2, 3, 4, 5
    errores_menores_a_1 = sum(abs(error) <= 1 for error in errores_totales)
    errores_menores_a_2 = sum(abs(error) <= 2 for error in errores_totales)
    errores_menores_a_3 = sum(abs(error) <= 3 for error in errores_totales)
    errores_menores_a_4 = sum(abs(error) <= 4 for error in errores_totales)
    errores_menores_a_5 = sum(abs(error) <= 5 for error in errores_totales)
    
    total_datos = len(errores_totales)

    # Calcular el porcentaje de errores menores a 1, 2, 3, 4, 5
    porcentaje_menores_a_1 = (errores_menores_a_1 / total_datos) * 100
    porcentaje_menores_a_2 = (errores_menores_a_2 / total_datos) * 100
    porcentaje_menores_a_3 = (errores_menores_a_3 / total_datos) * 100
    porcentaje_menores_a_4 = (errores_menores_a_4 / total_datos) * 100
    porcentaje_menores_a_5 = (errores_menores_a_5 / total_datos) * 100

    # Error máximo
    error_maximo = max(abs(error) for error in errores_totales)


    from sklearn.metrics import mean_absolute_percentage_error
    MAPE = mean_absolute_percentage_error(ytest, prediccionesTest)
    Accuracy = 1 - MAPE


    # Guardar resultados globales en un archivo de texto
    resultados_txt_path = os.path.join(carpeta, 'resultados.txt')
    with open(resultados_txt_path, 'a') as f:
        f.write("\n")
        f.write(f"Error promedio global: {error_promedio_global:.2f}\n")
        f.write(f"Desviación estándar global: {desviacion_estandar_global:.2f}\n")
        f.write(f"Error relativo porcentual promedio global: {error_relativo_porcentual_promedio:.2f}%\n")
        f.write(f"Cantidad de datos: {total_datos}\n")
        f.write(f"Cantidad de errores menores o iguales a 1: {errores_menores_a_1} ({porcentaje_menores_a_1:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 2: {errores_menores_a_2} ({porcentaje_menores_a_2:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 3: {errores_menores_a_3} ({porcentaje_menores_a_3:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 4: {errores_menores_a_4} ({porcentaje_menores_a_4:.2f}%)\n")
        f.write(f"Cantidad de errores menores o iguales a 5: {errores_menores_a_5} ({porcentaje_menores_a_5:.2f}%)\n")
        f.write(f"El error más grande cometido: {error_maximo:.2f}\n")
        f.write(f"el mape da: {MAPE}\n")
        f.write(f"La precisin es: {Accuracy}\n")

    # Guardar a un archivo CSV los resultados individuales
    resultados_csv_path = os.path.join(carpeta, 'resultados_predicciones.csv')
    df_resultados.to_csv(resultados_csv_path, index=False)


if __name__ == "__main__":
    setup_logger()

    # Pedir al usuario el nombre del modelo
    nombre_modelo = "modelo_1.2.20"  # Ejemplo de nombre
    carpeta, resultados_path = crear_carpeta_y_guardar(nombre_modelo)

    # Cargar el modelo previamente entrenado
    modelo = cargar_modelo("modelo_trained.keras")

    # Si el modelo se carga correctamente, proceder
    if modelo is not None:
        modelo.summary()
        scal = cargar_escaladores("scalers.pkl")
        if scal is not None:
            # Procesamiento de los datos
            df = cargar_datos()
            dias = [1, 2, 3, 4, 5]  # Ejemplo de días
            horas = [9, 10, 11, 12, 13, 14, 15, 16]  # Ejemplo de horas
            df = cargar_datos_especificos('potencias.csv', 'corrientes.csv', dias_semanales=dias, horas=horas)
            print("tengo estos datos ",df.shape)
            X, y = crear_ventana(df[9000:120000], 9*4, 4)

            


            inicio_train = 0
            fin_train = 10500
            inicio_val = fin_train+1
            fin_val = fin_train+1+4250
            inicio_test = fin_val+1
            fin_test = inicio_test+1+3000
            # conjunto de validación
            Xval = X[inicio_val:fin_val]
            yval = y[inicio_val:fin_val]
            #conjunto de entrenamiento
            Xtrain = X[inicio_train:fin_train]
            ytrain = y[inicio_train:fin_train]
            # conjunto de validación
            Xtest = X[inicio_test:fin_test]
            ytest = y[inicio_test:fin_test]

            from sklearn.preprocessing import MinMaxScaler
            scaleractiva = MinMaxScaler(feature_range=(0, 1))
            Xtrain_n = Xtrain.copy()
            Xtrain_n[:, :, 0] = scaleractiva.fit_transform(Xtrain[:, :, 0])
            Xval_n = Xval.copy()
            Xval_n[:, :, 0] = scaleractiva.transform(Xval[:, :, 0])
            Xtest_n = Xtest.copy()
            Xtest_n[:, :, 0] = scaleractiva.transform(Xtest[:, :, 0])
            print("la dimension es ", Xtest_n.shape)

            salidas = MinMaxScaler(feature_range=(0, 1))
            ytrain_n = ytrain.copy()
            ytrain_n = salidas.fit_transform(ytrain)
            yval_n = yval.copy()
            yval_n = salidas.transform(yval)
            ytest_n = salidas.transform(ytest)


            scalerdiferencias = MinMaxScaler(feature_range=(0, 1))
            Xtrain_n[:, :, 3] = scalerdiferencias.fit_transform(Xtrain[:, :, 3])
            Xval_n = Xval.copy()
            Xval_n[:, :, 3] = scalerdiferencias.transform(Xval[:, :, 3])
            Xtest_n = Xtest.copy()
            Xtest_n[:, :, 3] = scalerdiferencias.transform(Xtest[:, :, 3])

            scalermedicion = MinMaxScaler(feature_range=(0, 1))
            Xtrain_n[:, :, 4] =scalermedicion.fit_transform(Xtrain[:, :, 4])
            Xval_n = Xval.copy()
            Xval_n[:, :, 4] = scalermedicion.transform(Xval[:, :, 4])
            Xtest_n = Xtest.copy()
            Xtest_n[:, :, 4] = scalermedicion.transform(Xtest[:, :, 4])

            print(Xtest_n)


            scalers = {'scaleractiva': scaleractiva, 'salidas': salidas, 'scalerdiferencias': scalerdiferencias, 'scalermedicion': scalermedicion}
            #scalers = {'scaleractiva': scaleractiva, 'salidas': salidas, 'scalerdiferencias': scalerdiferencias}







            # Entrenar el modelo
            modell = entrenar_modelo(Xtrain_n, ytrain_n, Xval_n, yval_n)
            #modell = define_model(Xtrain_n, ytrain_n, Xval_n, yval_n)

            # Realizar predicciones
            prediccionesTest = modell.predict(Xtest_n)
            prediccionesTest = salidas.inverse_transform(prediccionesTest)

            # Calcular los resultados
            calcular_resultados(ytest, prediccionesTest,carpeta)

            # Guardar el modelo, los escaladores y los resultados
            guardar_modelo_y_resultados(carpeta, modell, scalers)

            print(f"Modelo, escaladores y resultados guardados en {carpeta}")
