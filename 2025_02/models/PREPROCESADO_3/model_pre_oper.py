#%% Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#%% Cargar datos
df = pd.read_csv('/home/ubuntu/STG-fractura_cadera/2025_02/models/PREPROCESADO_3/DATOS_PREPROCESADOS_3.csv')

# Selección de columnas relevantes para el modelo
columnas_relevantes = [
    'gidenpac', 'itipsexo', 'itipingr', 'ireingre', 'gsitalta', 'iotrocen', 'gdiagalt',
    'ds_izq_der', 'ds_turno', 'ds_edad', 'ds_estancia', 'ds_pre_oper', 'ds_post_oper',
    'ds_vivo_alta', 'ds_dia_semana_llegada', 'ds_mes_llegada', 'ds_centro_afueras',
    'ds_alergia_medicamentosa', 'ds_alergia_alimenticia', 'ds_otras_alergias',
    'movilidad', 'Barthel', 'braden', 'riesgo_caida', 'ds_ITU', 'ds_anemia',
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca',
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]
df = df[columnas_relevantes]

#%% Definir variables categóricas
cat_features = [
    'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 'gdiagalt',
    'ds_izq_der', 'ds_turno', 'ds_dia_semana_llegada', 'ds_mes_llegada',
    'ds_centro_afueras', 'ds_alergia_medicamentosa', 'ds_alergia_alimenticia',
    'ds_otras_alergias', 'movilidad', 'riesgo_caida', 'ds_ITU', 'ds_anemia',
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca',
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]

# Convertir columnas categóricas a string y rellenar valores nulos con "-999"
for col in cat_features:
    df[col] = df[col].astype(str).fillna("-999")

# Convertir las columnas numéricas (excluyendo la variable objetivo) a numérico
numeric_cols = [col for col in df.columns if col not in cat_features + ['ds_pre_oper']]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


#%% División de datos en predictores y variable objetivo
# Se asume que 'ds_pre_oper' es la variable a predecir (días de preoperatorio)
X = df.drop(['gsitalta', 'ds_vivo_alta', 'ds_estancia', 'ds_post_oper', 'ds_pre_oper'], axis=1)
y = df['ds_pre_oper']

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)

#%% Entrenar el modelo CatBoostRegressor con early stopping
model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    max_depth=7,
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=50
)

model.fit(X_train, y_train, cat_features=cat_features, 
          eval_set=(X_test, y_test), early_stopping_rounds=20)

#%% Evaluar el modelo en el conjunto de prueba
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)
print("R2 Score:", r2)

#%% Visualización: Valores Reales vs. Predichos
plt.figure(figsize=(8,6))
plt.scatter(y_test, preds, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Valor Real (días de preoperatorio)")
plt.ylabel("Valor Predicho")
plt.title("Comparación: Valores Reales vs. Predichos")
plt.show()

#%% Visualización: Análisis de Residuos
residuos = y_test - preds
plt.figure(figsize=(8,6))
plt.scatter(preds, residuos, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Valor Predicho")
plt.ylabel("Residuo (Real - Predicho)")
plt.title("Análisis de Residuos")
plt.show()

#%% Validación Cruzada (opcional)
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', 
                            fit_params={'cat_features': cat_features})
print("Scores de validación cruzada (R2):", cv_scores)
print("Media del R2 en validación cruzada:", cv_scores.mean())

#%% Modelo Base (Baseline): Predecir la media
baseline_pred = np.full_like(y_test, y_train.mean())
mse_baseline = mean_squared_error(y_test, baseline_pred)
r2_baseline = r2_score(y_test, baseline_pred)
print("Baseline - Mean Squared Error:", mse_baseline)
print("Baseline - R2 Score:", r2_baseline)

#%% Guardar el modelo entrenado
model.save_model("model_predict_preoper_days.cbm")

#%% Preparar nuevos datos para predicción
nuevos_datos = {
    'itipsexo': [0],
    'itipingr': [0],
    'ireingre': [0],
    'iotrocen': [1],
    'gdiagalt': ['S72.141A'],
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
    'ds_ITU': [0],
    'ds_anemia': [0],
    'ds_vitamina_d': [1],
    'ds_insuficiencia_respiratoria': [0],
    'ds_insuficiencia_cardiaca': [0],
    'ds_deterioro_cognitivo': [0],
    'ds_insuficiencia_renal': [0],
    'ds_HTA': [1],
    'ds_diabetes': [1]
}
nuevos_datos_df = pd.DataFrame(nuevos_datos)

# Convertir las columnas categóricas en nuevos datos a string y llenar valores nulos
for col in cat_features:
    nuevos_datos_df[col] = nuevos_datos_df[col].astype(str).fillna("-999")

# Convertir las demás columnas a numérico
for col in nuevos_datos_df.columns:
    if col not in cat_features:
        nuevos_datos_df[col] = pd.to_numeric(nuevos_datos_df[col], errors='coerce')

print("Tipos de datos en nuevos_datos_df:")
print(nuevos_datos_df.dtypes)

#%% Realizar predicción con nuevos datos
preds_new = model.predict(nuevos_datos_df)
print("Predicción de días de preoperatorio para nuevos datos:", preds_new)
