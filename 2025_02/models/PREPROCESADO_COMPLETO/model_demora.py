#%% Importar librerías
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#%% Cargar datos
df = pd.read_csv('/home/ubuntu/STG-fractura_cadera/2025_02/models/PREPROCESADO_COMPLETO/DATOS_PREPROCESADOS.csv')

# Selección de columnas relevantes para el modelo
columnas_relevantes = [
    'gsitalta', 'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 
    'gdiagalt', 'ds_izq_der', 'ds_turno', 'ds_edad', 'ds_estancia', 
    'ds_pre_oper', 'ds_post_oper', 'ds_vivo_alta', 'ds_dia_semana_llegada', 
    'ds_mes_llegada', 'ds_centro_afueras', 
    'ds_alergia_medicamentosa', 'ds_alergia_alimenticia', 'ds_otras_alergias', 
    'movilidad', 'Barthel', 'braden', 'riesgo_caida', 'ds_ITU', 'ds_anemia', 
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca', 
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]
df = df[columnas_relevantes]

#%% Definir variables categóricas
# Estas son las variables predictoras que son categóricas.
cat_features = [
    'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 'gdiagalt',
    'ds_izq_der', 'ds_turno', 'ds_dia_semana_llegada', 'ds_mes_llegada',
    'ds_centro_afueras', 'ds_alergia_medicamentosa', 'ds_alergia_alimenticia',
    'ds_otras_alergias', 'movilidad', 'riesgo_caida', 'ds_ITU', 'ds_anemia',
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca',
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]

# Convertir las variables categóricas a string (o category) y rellenar valores nulos con '-999'
for col in cat_features:
    df[col] = df[col].astype(str).fillna("-999")

# Si existen otras columnas que deben ser numéricas (por ejemplo, 'ds_edad', 'Barthel', 'braden'),
# se puede hacer una conversión (aquí dejamos las que no están en cat_features):
numeric_cols = [col for col in df.columns if col not in cat_features + ['ds_pre_oper']]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999)

#%% División de datos en predictores y variable objetivo
# Suponemos que 'ds_pre_oper' es la variable objetivo (número de días de preoperatorio)
# Se eliminan también algunas columnas que no se utilizarán para la predicción.
X = df.drop(['gsitalta', 'ds_vivo_alta', 'ds_estancia', 'ds_post_oper', 'ds_pre_oper'], axis=1)
y = df['ds_pre_oper']  # Variable objetivo numérica

# División en conjuntos de entrenamiento y prueba (en regresión no se usa stratify)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)

#%% Entrenar el modelo CatBoostRegressor
model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.1,
    max_depth=5,
    loss_function='RMSE',  # Función de pérdida para regresión
    eval_metric='RMSE',
    verbose=10
)

# Entrenar el modelo pasando la lista de columnas categóricas.
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

#%% Evaluar el modelo en el conjunto de prueba
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

#%% Guardar el modelo
model.save_model("model_2_ds_pre_oper.cbm")

#%% Preparar nuevos datos para predicción
# Asegurarse de que las claves del diccionario coincidan exactamente con las columnas de X.
nuevos_datos = {
    'itipsexo': [0],
    'itipingr': [0],
    'ireingre': [0],
    'iotrocen': [1],
    'gdiagalt': ['S72.141A'],  # Aquí se utiliza el valor categórico (string) esperado
    'ds_izq_der': [0],
    'ds_turno': [2],
    'ds_edad': [95],
    'ds_dia_semana_llegada': [5],
    'ds_mes_llegada': [9],
    'ds_centro_afueras': [1],
    'ds_alergia_medicamentosa': [1],
    'ds_alergia_alimenticia': [1],
    'ds_otras_alergias': [1],
    'movilidad': [2],
    'Barthel': [50],
    'braden': [13],
    'riesgo_caida': [7],
    'ds_ITU': [1],
    'ds_anemia': [0],
    'ds_vitamina_d': [0],
    'ds_insuficiencia_respiratoria': [0],
    'ds_insuficiencia_cardiaca': [0],
    'ds_deterioro_cognitivo': [0],
    'ds_insuficiencia_renal': [0],
    'ds_HTA': [1],
    'ds_diabetes': [1]
}

nuevos_datos_df = pd.DataFrame(nuevos_datos)

# Convertir las columnas categóricas a string en el nuevo dataset (rellenando valores nulos con '-999')
for col in cat_features:
    nuevos_datos_df[col] = nuevos_datos_df[col].astype(str).fillna("-999")

# Para las demás columnas numéricas (si las hubiera), asegurarse de que sean numéricas:
for col in nuevos_datos_df.columns:
    if col not in cat_features:
        nuevos_datos_df[col] = pd.to_numeric(nuevos_datos_df[col], errors='coerce').fillna(-999)

print("Tipos de datos en nuevos_datos_df:")
print(nuevos_datos_df.dtypes)

#%% Realizar predicciones con nuevos datos
preds_new = model.predict(nuevos_datos_df)
print("Predicción de días de preoperatorio para nuevos datos:", preds_new)

# %%
