import joblib
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('model_and_scalers1.h5')
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import numpy as np

# Cargar escaladores
with open('../scalers1.pkl', 'rb') as f:
    scalers = joblib.load(f)

# Verificar el contenido del archivo
print("Contenido del archivo 'scalers1.pkl':")
print(scalers)

# Verificar cada escalador
for name, scaler in scalers.items():
    print(f"\n{name}:")
    print("Tipo:", type(scaler))
    
    # Verificar atributos del escalador
    try:
        print("Mean:", scaler.mean_)
    except AttributeError:
        print("Mean attribute not found")
    
    try:
        print("Scale:", scaler.scale_)
    except AttributeError:
        print("Scale attribute not found")

    # Verificar si el escalador está ajustado
    try:
        scaler.mean_
        print(f"{name} está ajustado.")
    except AttributeError:
        print(f"{name} no está ajustado.")
    except NotFittedError:
        print(f"{name} no está ajustado.")

