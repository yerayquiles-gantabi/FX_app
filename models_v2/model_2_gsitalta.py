import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#%% Cargar datos
df = pd.read_csv('/home/ubuntu/STG-fractura_cadera/models_v2/DATOS_PREPROCESADOS.csv')

# Selección de todas las columnas relevantes para el modelo
df = df[['gsitalta', 'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 
         'gdiagalt', 'ds_izq_der', 'ds_turno', 'ds_edad', 'ds_estancia', 
         'ds_pre_oper', 'ds_post_oper', 'ds_vivo_alta', 'ds_dia_semana_llegada', 
         'ds_mes_llegada', 'ds_centro_afueras', 'ntensmin', 'ntensmax', 
         'ntempera', 'nsatuoxi', 'ds_alergia_medicamentosa', 
         'ds_alergia_alimenticia', 'ds_otras_alergias', 'movilidad', 
         'lugar_residencia', 'lugar_procedencia', 'destino_alta', 'Barthel', 
         'braden', 'riesgo_caida', 'ds_ITU', 'ds_anemia', 'ds_vitamina_d', 
         'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca', 
         'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes']]

#%% Conversión de variables categóricas
cat_features = ['itipsexo', 'itipingr', 'ireingre', 'iotrocen', 
                'gdiagalt', 'ds_izq_der', 'ds_turno', 
                'ds_dia_semana_llegada', 'ds_mes_llegada', 'ds_centro_afueras', 
                'ds_alergia_medicamentosa', 'ds_alergia_alimenticia', 
                'ds_otras_alergias', 'movilidad', 'lugar_residencia', 
                'lugar_procedencia', 'riesgo_caida', 'ds_ITU', 'ds_anemia', 
                'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 
                'ds_insuficiencia_cardiaca', 'ds_deterioro_cognitivo', 
                'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes']

for col in cat_features:
    df[col] = df[col].astype('category')

#%% División de datos en entrenamiento y prueba
X = df.drop(['gsitalta', 'ds_estancia', 'ds_vivo_alta', 'ds_pre_oper', 'ds_post_oper', 'destino_alta'], axis=1)
y = df['gsitalta']  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Entrenar el modelo CatBoostClassifier
model = CatBoostClassifier(iterations=200,
                           learning_rate=0.1,
                           max_depth=5,
                           loss_function='MultiClass',
                           eval_metric='Accuracy',
                           verbose=10)

model.fit(X_train, y_train, cat_features=cat_features, eval_set=(X_test, y_test))

#%% Guardar modelo
model.save_model("model_2_gsitalta.cbm")

#%% Nuevos datos
nuevos_datos = {
    'itipsexo': [1],
    'itipingr': [0],
    'ireingre': [0],
    'iotrocen': [1],
    'gdiagalt': ['M84.459A'],
    'ds_izq_der': [0],
    'ds_turno': [1],
    'ds_edad': [72],
    'ds_dia_semana_llegada': [5],
    'ds_mes_llegada': [9],
    'ds_centro_afueras': [1],
    'ntensmin': [55],
    'ntensmax': [100],
    'ntempera': [36.7],
    'nsatuoxi': [75],
    'ds_alergia_medicamentosa': [0],
    'ds_alergia_alimenticia': [0],
    'ds_otras_alergias': [0],
    'movilidad': [2],
    'lugar_residencia': [-999],
    'lugar_procedencia': [-999],
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
    'ds_HTA': [0],
    'ds_diabetes': [1]
}

nuevos_datos_df = pd.DataFrame(nuevos_datos)

# Verificar valores NaN y convertir a texto en columnas categóricas
for col in cat_features:
    nuevos_datos_df[col] = nuevos_datos_df[col].astype(str).replace('nan', 'NaN')  # Convertir NaN a cadena

# Verificar que todos los valores categóricos son cadenas
print(nuevos_datos_df[cat_features].dtypes)

# Realizar predicciones
preds = model.predict(nuevos_datos_df)
preds_proba = model.predict_proba(nuevos_datos_df)

print("Clase Predicha:", preds)
print("Probabilidades por clase:", preds_proba)

