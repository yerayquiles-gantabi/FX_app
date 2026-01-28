'''
Transforma los códigos numéricos del JSON en texto legible para Streamlit.
Regla: Las nuevas variables se llaman {variable_original}_map

Tambien hace la inversa: convierte datos del simulador (con sufijo _map) al formato que espera el modelo (sin sufijo).
'''
import pandas as pd

def enriquecer_datos_para_ui(data):
    """
    Recibe el diccionario de datos (data) cargado del JSON.
    Devuelve el mismo diccionario con campos extra '_map' añadidos.
    """
    if not data: return {}
    
    # 1. Diccionarios de Mapeo
    MAPPINGS = {
        "sexo": {0: "Hombre", 1: "Mujer"},
        "si_no": {0: "No", 1: "Sí"},
        "lado": {0: "No especificado", 1: "Izquierda", 2: "Derecha"},
        "procedencia": {0: "Domicilio", 1: "Otro Centro/Hospital"},
        "residencia": {1: "Urbano (León)", 0: "Rural/Afueras"},
        "turno": {0: "Mañana", 1: "Tarde", 2: "Noche", None: "Desconocido"}
    }

    # 2. Generación de Textos (Mapeo directo nombre_original -> nombre_original_map)
    
    # Demográficos
    data["itipsexo_map"] = MAPPINGS["sexo"].get(data.get("itipsexo"), "Desconocido")
    data["ds_edad_map"] = f"{data.get('ds_edad', 0)} años"
    data["iotrocen_map"] = MAPPINGS["procedencia"].get(data.get("iotrocen"), "Desconocida")
    data["ds_centro_afueras_map"] = MAPPINGS["residencia"].get(data.get("ds_centro_afueras"), "Desconocida")
    
    # -------------------------------------------------------------------------
    # FECHAS (Combinar Fecha + Hora)
    # -------------------------------------------------------------------------
    fecha = data.get("fllegada")
    hora = data.get("hllegada")
    
    if fecha and hora:
        # Si tenemos fecha y hora separadas en el JSON
        texto_completo = f"{fecha} {hora}" # Ej: "2026-01-02 17:33:20"
        try:
            # Lo convertimos a formato fecha real para formatearlo bonito
            data["fllegada_map"] = pd.to_datetime(texto_completo).strftime("%d/%m/%Y %H:%M")
        except:
            # Si falla el formato, lo mostramos tal cual viene
            data["fllegada_map"] = texto_completo
            
    elif fecha:
        # Si solo tenemos la fecha
        try:
            data["fllegada_map"] = pd.to_datetime(fecha).strftime("%d/%m/%Y")
        except:
            data["fllegada_map"] = str(fecha)
            
    else:
        # Fallback para versiones antiguas del JSON
        raw = data.get("fllegada_raw")
        if raw:
            try:
                data["fllegada_map"] = pd.to_datetime(raw).strftime("%d/%m/%Y %H:%M")
            except:
                data["fllegada_map"] = str(raw)
        else:
            data["fllegada_map"] = "Desconocida"
    # Turno
    data["turno_raw_map"] = MAPPINGS["turno"].get(data.get("turno_raw"), "Desconocido")
    
    # Diagnóstico Lado
    data["ds_izq_der_map"] = MAPPINGS["lado"].get(data.get("ds_izq_der"), "Desconocido")

    # Clínicos (Sí/No)
    listado_clinico = [
        "ds_HTA", "ds_diabetes", "ds_deterioro_cognitivo", 
        "ds_insuficiencia_respiratoria", "ds_insuficiencia_cardiaca", 
        "ds_anemia", "ds_insuficiencia_renal", 
        "ds_ITU", "ds_vitamina_d", "ds_osteoporosis","ds_alergia_medicamentosa",
        "ds_alergia_alimentaria", "ds_otra_alergias","ds_obesidad", "ds_acido_folico"
    ]
    
    for key in listado_clinico:
        data[f"{key}_map"] = MAPPINGS["si_no"].get(data.get(key), "No")

    # Escalas con descripción
    # Barthel
    try:
        b = int(float(data.get('barthel') or 0)) 
    except (ValueError, TypeError):
        b = 0
    dep = 'Total' if b<20 else 'Grave' if b<60 else 'Leve/Indep'
    data["barthel_map"] = f"{b}"
    
    # Movilidad
    try:
        m = int(float(data.get('movilidad') or 0))
    except (ValueError, TypeError):
        m = 0
    data["movilidad_map"] = f"{m} "

    # Braden
    try:
        br = int(float(data.get('braden') or 0))
    except: br = 0
    data["braden_map"] = str(br)

    # Riesgo Caida
    try:
        rc = int(float(data.get('riesgo_caida') or 0))
    except: rc = 0
    data["riesgo_caida_map"] = str(rc)

    # Códigos CIE (gdiagalt)
    codigos_activos = []
    prefix = "gdiagalt_"
    for key, value in data.items():
        if key.startswith(prefix) and (value == 1 or value == "1"):
            codigos_activos.append(key.replace(prefix, ""))
    data["gdiagalt_map"] = ", ".join(codigos_activos) if codigos_activos else "Sin códigos"

    # -------------------------------------------------------------------------
    # SITUACIÓN AL ALTA (gsitalta) -> Mejora / Empeora
    # -------------------------------------------------------------------------
    try:
        val_gsi = int(float(data.get("gsitalta") or 0))
    except: 
        val_gsi = 0
        
    if val_gsi in [1, 2]:
        data["gsitalta_map"] = "Mejora"
    elif val_gsi in [3, 4, 5]:
        data["gsitalta_map"] = "Empeora"
    else:
        data["gsitalta_map"] = "Desconocido"

    # -------------------------------------------------------------------------
    # FORMATEO DE CONSTANTES VITALES
    # -------------------------------------------------------------------------
    
    # 1. Enteros (Tensión, Saturación) - Quitar decimales
    vars_enteras = ["ntensmin", "ntensmax", "nsatuoxi"]
    
    for v in vars_enteras:
        valor_raw = data.get(v)
        try:
            if valor_raw is not None:
                data[f"{v}_map"] = str(int(float(valor_raw)))
            else:
                data[f"{v}_map"] = "N/A"
        except (ValueError, TypeError):
            data[f"{v}_map"] = str(valor_raw) if valor_raw else "N/A"

    # 2. Decimales (Temperatura) - Forzar 1 decimal
    try:
        temp = data.get("ntempera")
        if temp is not None:
            data["ntempera_map"] = f"{float(temp):.1f}" # Fuerza "36.5"
        else:
            data["ntempera_map"] = "N/A"
    except:
        data["ntempera_map"] = str(data.get("ntempera", "N/A"))

    return data

def preparar_datos_simulacion_para_modelo(datos_simulacion, ruta_json_base="paciente_SRRD193407690.json"):
    """
    Convierte datos del simulador (con sufijo _map) al formato que espera el modelo (sin sufijo).
    También copia las columnas one-hot encoded del paciente base.
    
    Args:
        datos_simulacion: Diccionario con datos del formulario del simulador (con sufijo _map)
        ruta_json_base: Ruta al JSON del paciente real para copiar estructura one-hot
    
    Returns:
        Diccionario con datos en formato del modelo
    """
    import json
    import os
    
    # Cargar datos base del paciente real para obtener estructura completa
    try:
        # Intentar ruta relativa desde app.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_completa = os.path.join(base_dir, ruta_json_base)
        with open(ruta_completa, "r") as file:
            data_base = json.load(file)
    except:
        # Fallback: intentar desde directorio actual
        with open(ruta_json_base, "r") as file:
            data_base = json.load(file)
    
    # Crear diccionario de salida
    datos_modelo = {}
    
    # Mapeo de nombres: simulador (_map) -> modelo (sin _map)
    mapeo = {
        'itipsexo_map': 'itipsexo',
        'ds_edad_map': 'ds_edad',
        'iotrocen_map': 'iotrocen',
        'ds_centro_afueras_map': 'ds_centro_afueras',
        'ntensmin_map': 'ntensmin',
        'ntensmax_map': 'ntensmax',
        'ntempera_map': 'ntempera',
        'nsatuoxi_map': 'nsatuoxi',
        'ds_ITU_map': 'ds_ITU',
        'ds_insuficiencia_respiratoria_map': 'ds_insuficiencia_respiratoria',
        'ds_insuficiencia_cardiaca_map': 'ds_insuficiencia_cardiaca',
        'ds_deterioro_cognitivo_map': 'ds_deterioro_cognitivo',
        'ds_insuficiencia_renal_map': 'ds_insuficiencia_renal',
        'ds_HTA_map': 'ds_HTA',
        'ds_diabetes_map': 'ds_diabetes',
        'ds_obesidad_map': 'ds_obesidad',
        'ds_osteoporosis_map': 'ds_osteoporosis',
        'ds_anemia_map': 'ds_anemia',
        'ds_vitamina_d_map': 'ds_vitamina_d',
        'ds_acido_folico_map': 'ds_acido_folico',
        'ds_alergia_medicamentosa_map': 'ds_alergia_medicamentosa',
        'ds_alergia_alimentaria_map': 'ds_alergia_alimenticia',  # OJO: alimentaria -> alimenticia
        'ds_otra_alergias_map': 'ds_otras_alergias',  # OJO: otra -> otras
        'barthel_map': 'Barthel',  # OJO: mayúscula B
        'braden_map': 'braden',
        'riesgo_caida_map': 'riesgo_caida',
        'movilidad_map': 'movilidad',
        'ds_izq_der_map': 'ds_izq_der',
        'gdiagalt_map': 'gdiagalt'
    }
    
    # Aplicar mapeo básico
    for key_simulador, key_modelo in mapeo.items():
        if key_simulador in datos_simulacion:
            datos_modelo[key_modelo] = datos_simulacion[key_simulador]
    
    # Copiar todas las columnas one-hot encoded del paciente base
    # (necesarias para el modelo pero no editables en el simulador)
    for key in data_base.keys():
        if key.startswith('gdiagalt_') or \
           key.startswith('ds_izq_der_') or \
           key.startswith('ds_dia_semana_') or \
           key.startswith('ds_mes_') or \
           key.startswith('ds_turno_'):
            datos_modelo[key] = data_base[key]
    
    return datos_modelo