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
            dias = [0,1,2,3,4,5,6,7]  # Lunes, Miércoles, Viernes
            horas = [1,2,3,4,5,6,7,8,9, 10, 11, 12,13,14,15,16,17,18,19,20,21,22,23]  # De 9 a 12 horas
            df = cargar_datos_especificos('potencias.csv', 'corrientes.csv', dias_semanales=dias, horas=horas)
            #df = codificar_tiempo(df)
            #X= crear_ventana_dataset(df,4)
            print(df.shape)
            X, y = crear_ventana(df[40000:120000], 4,1)
            
            inicio_train = 0
            fin_train = 25000
            inicio_val = fin_train+1
            fin_val = fin_train+1+10000
            inicio_test = fin_val+1
            fin_test = inicio_test+1+2000
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
            print(Xtest_n)
            print("la dimension es ", Xtest_n.shape)

            salidas = MinMaxScaler(feature_range=(0, 1))
            ytrain_n = ytrain.copy()
            ytrain_n = salidas.fit_transform(ytrain)
            yval_n = yval.copy()
            yval_n = salidas.transform(yval)
            ytest_n = salidas.transform(ytest)

            
            import pickle
            scalers = {'scaleractiva': scaleractiva, 'salidas': salidas}

            with open('scalers.pkl', 'wb') as f:
                pickle.dump(scalers, f)

            print("media ", scaleractiva.data_min_)
            print("desv ", scaleractiva.data_max_)
            #X_n = escalar_entrada(X,scal)
            from sklearn.utils import shuffle

            #Xtrain_n, ytrain_n = shuffle(Xtrain_n, ytrain_n, random_state=0)
            #Xval_n, yval_n = shuffle(Xval_n, yval_n, random_state=0)


            
            logging.info("inicio prediccion")
            #predicciones = modelo.predict(X_n, batch_size=1)



            modell = entrenar_modelo(Xtrain_n,ytrain_n, Xval_n, yval_n)
            #modell = entrenar_modelo_con_atencion(Xtrain_n,ytrain_n, Xval_n, yval_n)
            

            modell.save('modelo')
            print("fin prediccion")
            #predicciones = modelo.predict(X_n, batch_size=1)

            #predicciones = scal['salidas'].inverse_transform(predicciones)
            #print(predicciones)
            #escribir_bd(predicciones)




            prediccionesTest = modell.predict(Xtrain_n)  # Esto sería lo que se debe hacer si aún no se ha calculado
            prediccionesTest = salidas.inverse_transform(prediccionesTest)
            
            print("Tamaño de prediccionesTest:", len(prediccionesTest))
            print("Tamaño de yTest:", len(ytest))

            print("fin prediccion")
            #######################3
            #### VOLVER A PONER ESTO CUANDO NO PREDIGA CON LOS DE TRAIN
            #yTest = ytest
            yTest = ytrain

            import pandas as pd

            # Crear listas para almacenar los resultados
            valores_reales = []
            predicciones = []
            errores = []
            resultados = []
            errores_totales = []

            import numpy as np

            # Supongamos que df tiene los datos originales con las columnas codificadas
            for valor in range(len(yTest)):
                y_real = yTest[valor]
                prediccion = prediccionesTest[valor]
                
                # Calcular errores
                errores = [prediccion[i] - y_real[i] for i in range(len(y_real))]
                errores_acumulativos = []
                
                # Calcular error promedio y desviación estándar para este valor
                error_promedio = np.mean(np.abs(errores))
                desviacion_estandar = np.std(errores)

                # Contar cuántas diferencias son mayores a 3 kilowatts
                diferencias_mayores_a_3 = sum(abs(error) > 3 for error in errores)
                
                # Obtener los valores de codificación para cada predicción
                for i in range(len(y_real)):
                    errores_acumulativos.append(np.abs(errores[i]))
                    
                    # Calcular el error promedio acumulativo hasta el índice actual
                    error_promedio_acumulativo = np.mean(errores_acumulativos)
                    desviacion_estandar_acumulativa = np.std(errores_acumulativos)
                    
                    # Calcular el error relativo porcentual
                    if y_real[i] != 0:  # Evitar división por cero
                        error_relativo_porcentual = (abs(errores[i]) / abs(y_real[i])) * 100
                    else:
                        error_relativo_porcentual = 0  # Si y_real es cero, podemos definir el error relativo como 0

                    # Almacenar resultados
                    resultados.append({
                        'valor': valor,
                        'prediccion': prediccion[i],
                        'valor_real': y_real[i],
                        'error': errores[i],
                        'error_promedio': error_promedio,
                        'desviacion_estandar': desviacion_estandar,
                        'diferencias_mayores_a_3': diferencias_mayores_a_3,
                        'error_promedio_acumulativo': error_promedio_acumulativo,
                        'desviacion_estandar_acumulativa': desviacion_estandar_acumulativa,
                        'error_relativo_porcentual': error_relativo_porcentual  # Nueva columna
                    })

            # Convertir a DataFrame
            df_resultados = pd.DataFrame(resultados)

            # Calcular error relativo porcentual promedio global
            error_relativo_porcentual_promedio = np.mean(df_resultados['error_relativo_porcentual'])

            # Calcular error promedio global y desviación estándar promedio
            error_promedio_global = np.mean(df_resultados['error'])
            desviacion_estandar_global = np.std(df_resultados['error'])

            # Guardar a un archivo CSV
            df_resultados.to_csv('resultados_predicciones_con_datos.csv', index=False)

            # Imprimir resultados globales
            print(f"Error promedio global: {error_promedio_global:.2f}")
            print(f"Desviación estándar global: {desviacion_estandar_global:.2f}")
            print(f"Error relativo porcentual promedio global: {error_relativo_porcentual_promedio:.2f}%")
            print("Resultados guardados en 'resultados_predicciones_con_datos.csv'")




