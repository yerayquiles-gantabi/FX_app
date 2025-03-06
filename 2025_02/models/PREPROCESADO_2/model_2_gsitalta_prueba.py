#%% Importar librerías
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTENC
from collections import Counter

#%% Cargar datos
df = pd.read_csv('/home/ubuntu/STG-fractura_cadera/2025_02/models/PREPROCESADO_2/DATOS_PREPROCESADOS_2.csv')

# Selección de columnas relevantes
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

#%% Variables categóricas
cat_features = [
    'itipsexo', 'itipingr', 'ireingre', 'iotrocen', 'gdiagalt', 
    'ds_izq_der', 'ds_turno', 'ds_dia_semana_llegada', 'ds_mes_llegada', 
    'ds_centro_afueras', 'ds_alergia_medicamentosa', 'ds_alergia_alimenticia', 
    'ds_otras_alergias', 'ds_ITU', 'ds_anemia', 
    'ds_vitamina_d', 'ds_insuficiencia_respiratoria', 'ds_insuficiencia_cardiaca', 
    'ds_deterioro_cognitivo', 'ds_insuficiencia_renal', 'ds_HTA', 'ds_diabetes'
]

# Convertir variables categóricas a tipo `str`
for col in cat_features:
    df[col] = df[col].astype(str)

#%% 🔹 Eliminar clases con menos de 3 muestras ANTES de train_test_split
class_counts = df['gsitalta'].value_counts()
valid_classes = class_counts[class_counts >= 3].index
df_filtered = df[df['gsitalta'].isin(valid_classes)]

#%% Separar variable objetivo y predictores
X = df_filtered.drop(['gsitalta', 'ds_vivo_alta', 'ds_estancia', 'ds_pre_oper', 'ds_post_oper'], axis=1)
y = df_filtered['gsitalta']

# División de datos en entrenamiento y prueba con `stratify`
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 🔹 Verificar si después de train_test_split sigue habiendo clases con menos de 3 muestras
class_counts_train = y_train.value_counts()
valid_classes_train = class_counts_train[class_counts_train >= 3].index
X_train_filtered = X_train[y_train.isin(valid_classes_train)]
y_train_filtered = y_train[y_train.isin(valid_classes_train)]

print("Distribución después de eliminar clases con menos de 3 muestras:", Counter(y_train_filtered))

#%% 🔹 Aplicar balanceo con SMOTENC si hay al menos 2 clases
if len(set(y_train_filtered)) < 2:
    print("❌ No se puede aplicar SMOTENC porque hay menos de 2 clases.")
    X_train_balanced, y_train_balanced = X_train_filtered, y_train_filtered
else:
    # Obtener índices de variables categóricas
    cat_feature_indices = [X_train_filtered.columns.get_loc(col) for col in cat_features if col in X_train_filtered.columns]

    # Aplicar SMOTENC para balanceo de clases (evitando sobreajuste en clases mayoritarias)
    smote_nc = SMOTENC(categorical_features=cat_feature_indices, random_state=42, k_neighbors=1, sampling_strategy="not majority")
    X_train_balanced, y_train_balanced = smote_nc.fit_resample(X_train_filtered, y_train_filtered)

    print("Distribución balanceada de clases:", Counter(y_train_balanced))

#%% 🔹 Asegurar que las variables categóricas están bien definidas
for col in cat_features:
    X_train_balanced[col] = X_train_balanced[col].astype(str)
    X_test[col] = X_test[col].astype(str)

#%% 🔹 Detectar si el problema es **binario** o **multiclase**
num_classes = len(set(y_train_balanced))

if num_classes == 2:
    loss_function = "LogLoss"  # Clasificación binaria
    eval_metric = "Accuracy"
else:
    loss_function = "MultiClass"  # Se usa MultiClass en vez de OneVsAll
    eval_metric = "TotalF1"  # F1-score para multiclase

print(f"🔍 El problema tiene {num_classes} clases. Usando loss_function={loss_function}")

#%% Entrenar el modelo con la función de pérdida correcta
model = CatBoostClassifier(
    iterations=500,                
    learning_rate=0.05,            
    max_depth=6,                   
    loss_function=loss_function,    # Se ajusta automáticamente
    eval_metric=eval_metric,       
    verbose=10
)

model.fit(X_train_balanced, y_train_balanced, cat_features=cat_features, eval_set=(X_test, y_test))

#%% Predicciones y métricas
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# %%
