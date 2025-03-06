#%% Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#%% Cargar datos
df = pd.read_csv('/home/ubuntu/STG-fractura_cadera/2025_02/models/PREPROCESADO_2/DATOS_PREPROCESADOS_2.csv')

# Selección de columnas relevantes para el modelo
columnas_relevantes = [
    'gsitalta', 'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 
    'gdiagalt', 'ds_izq_der', 'ds_turno', 'ds_edad', 'ds_estancia', 
    'ds_pre_oper', 'ds_post_oper', 'ds_vivo_alta', 'ds_dia_semana_llegada', 
    'ds_mes_llegada', 'ds_centro_afueras', 
    'ds_alergia_medicamentosa', 'ds_alergia_alimenticia', 'ds_otras_alergias', 
    'ds_ITU', 'ds_anemia', 
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca', 
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]
df = df[columnas_relevantes]

#%% Definir variables categóricas
cat_features = [
    'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 'gdiagalt',
    'ds_izq_der', 'ds_turno', 'ds_dia_semana_llegada', 'ds_mes_llegada',
    'ds_centro_afueras', 'ds_alergia_medicamentosa', 'ds_alergia_alimenticia',
    'ds_otras_alergias', 'ds_ITU', 'ds_anemia',
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca',
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]


# Convertir columnas numéricas (no categóricas ni la variable objetivo) a numérico
numeric_cols = [col for col in df.columns if col not in cat_features + ['ds_pre_oper']]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999)

#%% División de datos en predictores y variable objetivo
X = df.drop(['gsitalta', 'ds_vivo_alta', 'ds_estancia', 'ds_post_oper', 'ds_pre_oper'], axis=1)
y = df['ds_pre_oper']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dimensiones de X_train:", X_train.shape)
print("Dimensiones de X_test:", X_test.shape)

#%% Búsqueda de hiperparámetros con GridSearchCV
# Definimos un diccionario de hiperparámetros a explorar
param_grid = {
    'iterations': [500, 1000],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5, 7, 9]
}

# Creamos el modelo base (sin early stopping, se lo agregaremos en la llamada a fit)
model = CatBoostRegressor(
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=0  # sin salida durante GridSearch
)

# Configuramos GridSearchCV sin 'fit_params'
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

# Ahora pasamos los parámetros adicionales directamente en fit()
grid_search.fit(X_train, y_train, cat_features=cat_features, early_stopping_rounds=20, eval_set=(X_test, y_test))
print("Mejores parámetros:", grid_search.best_params_)
print("Mejor R2 en validación:", grid_search.best_score_)


#%% Entrenar el modelo final con los mejores parámetros y early stopping
best_params = grid_search.best_params_
final_model = CatBoostRegressor(
    iterations=best_params['iterations'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    loss_function='RMSE',
    eval_metric='RMSE',
    verbose=10
)

final_model.fit(X_train, y_train, cat_features=cat_features, 
                early_stopping_rounds=20, eval_set=(X_test, y_test))

#%% Evaluar el modelo final en el conjunto de prueba
preds = final_model.predict(X_test)
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
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Valor Real")
plt.ylabel("Valor Predicho")
plt.title("Valores Reales vs. Predichos")
plt.show()

#%% Visualización: Residuos
residuos = y_test - preds
plt.figure(figsize=(8,6))
plt.scatter(preds, residuos, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Valor Predicho")
plt.ylabel("Residuo (Real - Predicho)")
plt.title("Análisis de Residuos")
plt.show()

#%% Validación Cruzada con el modelo final
cv_scores = cross_val_score(final_model, X, y, cv=5, scoring='r2', fit_params={'cat_features': cat_features})
print("Scores de validación cruzada (R2):", cv_scores)
print("Media del R2 en validación cruzada:", cv_scores.mean())

#%% Modelo Base (Baseline): Predecir la media
baseline_pred = np.full_like(y_test, y_train.mean())
mse_baseline = mean_squared_error(y_test, baseline_pred)
r2_baseline = r2_score(y_test, baseline_pred)
print("Baseline - Mean Squared Error:", mse_baseline)
print("Baseline - R2 Score:", r2_baseline)

#%% Guardar el modelo final
final_model.save_model("/home/ubuntu/STG-fractura_cadera/models_v2/PREPROCESADO_2/model_2_ds_pre_oper.cbm")

#%% Preparar nuevos datos para predicción
nuevos_datos = {
    'itipsexo': [1],
    'itipingr': [0],
    'ireingre': [1],
    'iotrocen': [1],
    'gdiagalt': ['S72.141A'],
    'ds_izq_der': [0],
    'ds_turno': [2],
    'ds_edad': [80],
    'ds_dia_semana_llegada': [5],
    'ds_mes_llegada': [9],
    'ds_centro_afueras': [1],
    'ds_alergia_medicamentosa': [1],
    'ds_alergia_alimenticia': [1],
    'ds_otras_alergias': [1],
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

# Convertir las columnas categóricas a string en el nuevo dataset
for col in cat_features:
    nuevos_datos_df[col] = nuevos_datos_df[col].astype(str).fillna("-999")

# Para las demás columnas numéricas, asegurar la conversión
for col in nuevos_datos_df.columns:
    if col not in cat_features:
        nuevos_datos_df[col] = pd.to_numeric(nuevos_datos_df[col], errors='coerce').fillna(-999)

print("Tipos de datos en nuevos_datos_df:")
print(nuevos_datos_df.dtypes)

#%% Realizar predicciones con nuevos datos
preds_new = final_model.predict(nuevos_datos_df)
print("Predicción de días de preoperatorio para nuevos datos:", preds_new)
