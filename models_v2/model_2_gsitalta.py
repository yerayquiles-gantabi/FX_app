#%%
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#%% Cargar datos
df = pd.read_csv('/home/ubuntu/STG-fractura_cadera/models_v2/DATOS_PREPROCESADOS.csv')

# Selección de columnas relevantes para el modelo
columnas_relevantes = [
    'gsitalta', 'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 
    'gdiagalt', 'ds_izq_der', 'ds_turno', 'ds_edad', 'ds_estancia', 
    'ds_pre_oper', 'ds_post_oper', 'ds_vivo_alta', 'ds_dia_semana_llegada', 
    'ds_mes_llegada', 'ds_centro_afueras', 
    'ds_alergia_medicamentosa','ds_alergia_alimenticia', 'ds_otras_alergias', 
    'movilidad', 'Barthel', 'braden', 'riesgo_caida', 'ds_ITU', 'ds_anemia', 
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca', 
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]
df = df[columnas_relevantes]

#%% Definir variables categóricas
# Lista de columnas que se tratarán como categóricas en CatBoost
cat_features = [
    'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 'gdiagalt', 
    'ds_izq_der', 'ds_turno', 'ds_dia_semana_llegada', 'ds_mes_llegada', 
    'ds_centro_afueras', 'ds_alergia_medicamentosa', 'ds_alergia_alimenticia', 
    'ds_otras_alergias', 'movilidad', 'riesgo_caida', 'ds_ITU', 'ds_anemia', 
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca', 
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]

# Convertir a tipo category inicialmente (esto puede ayudar para el análisis)
for col in cat_features:
    df[col] = df[col].astype('category')

#%% División de datos en entrenamiento y prueba
# Se eliminan columnas que no serán usadas como predictores
X = df.drop(['gsitalta', 'ds_vivo_alta', 'ds_estancia', 'ds_pre_oper', 'ds_post_oper'], axis=1)
y = df['gsitalta']  # Variable objetivo

# IMPORTANT: CatBoost requiere que las variables categóricas tengan valores de tipo entero o cadena.
# Convertimos todas las columnas categóricas a cadena. Si hay valores NaN, se reemplazan por la cadena '-999'
for col in cat_features:
    # Se aplica sobre X, ya que es el conjunto que se usa para entrenar
    X[col] = X[col].apply(lambda x: str(x) if pd.notnull(x) else '-999')

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Entrenar el modelo CatBoostClassifier
model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.1,
    max_depth=5,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    verbose=10
)

# Se pasa la lista de nombres de columnas categóricas para que CatBoost las procese correctamente.
model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

#%%
preds = model.predict(X_test)
print(classification_report(y_test, preds))

#%% Guardar modelo
model.save_model("model_2_gsitalta.cbm")

#%% Preparar nuevos datos para predicción
# Se debe asegurar que las columnas de nuevos_datos coincidan exactamente con las de X
nuevos_datos = {
    'itipsexo': [1],
    'itipingr': [0],
    'ireingre': [0],
    'iotrocen': [1],
    'gdiagalt': ['S72.141A'],
    'ds_izq_der': [0],
    'ds_turno': [1],
    'ds_edad': [100],
    'ds_dia_semana_llegada': [5],
    'ds_mes_llegada': [9],
    'ds_centro_afueras': [1],
    'ds_alergia_medicamentosa': [1],
    'ds_alergia_alimenticia': [1],
    'ds_otras_alergias': [1],
    'movilidad': [2],
    'Barthel': [-999],
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

# Convertir a cadena las columnas categóricas en el dataframe de nuevos datos,
# reemplazando valores nulos (NaN) por la cadena '-999'
for col in cat_features:
    nuevos_datos_df[col] = nuevos_datos_df[col].apply(lambda x: str(x) if pd.notnull(x) else '-999')

# Verificar que todos los valores en columnas categóricas sean cadenas
print("Tipos de datos en columnas categóricas de nuevos_datos_df:")
print(nuevos_datos_df[cat_features].dtypes)

#%% Realizar predicciones
preds = model.predict(nuevos_datos_df)
preds_proba = model.predict_proba(nuevos_datos_df)

print("Clase Predicha:", preds)
print("Probabilidades por clase:", preds_proba)

# %%
