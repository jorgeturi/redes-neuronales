import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

# Crear y guardar el modelo
model = Sequential([
    Dense(10, activation='relu', input_shape=(10,)),
    Dense(1, activation='sigmoid')
])
model.save('model_and_scalers1.h5')

# Crear y guardar los escaladores
scaleractiva = StandardScaler()
scalercosdia = StandardScaler()
scalersendia = StandardScaler()
scalercoshora = StandardScaler()
scalersenhora = StandardScaler()
scaler_y = StandardScaler()

# Ajustar escaladores (ejemplo)
# scaleractiva.fit(data_activa)
# ...

# Guardar escaladores
with open('scalers1.pkl', 'wb') as f:
    joblib.dump({
        'scaleractiva': scaleractiva,
        'scalercosdia': scalercosdia,
        'scalersendia': scalersendia,
        'scalercoshora': scalercoshora,
        'scalersenhora': scalersenhora,
        'scaler_y': scaler_y
    }, f)