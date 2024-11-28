import pandas as pd  
import plotly.express as px  
import streamlit as st 
import streamlit as st
import json

# Cargar el archivo CSS
def load_css(file_path):
    with open(file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Llamar a la función para cargar estilos personalizados
load_css("custom_styles.css")

# ___________________________________________________ VARIABLES _________________________________________
# _______________________________________________________________________________________________________
# Cargar datos desde el archivo JSON
with open("data_paciente.json", "r") as file:
    data = json.load(file)

# Asignar variables desde el JSON
id_paciente = data["id_paciente"]
sexo = data["sexo"]
edad = data["edad"]
lugar_residencia = data["lugar_residencia"]
tipo_fractura = data["tipo_fractura"]
lado_fractura = data["lado_fractura"]
resultado_demora = data["resultado_demora"]
predict_estancia = data["predict_estancia"]
postoperatorio = data["postoperatorio"]
postoperatorio_fijo = data["postoperatorio_fijo"]
prediccion_costes = data["prediccion_costes"]
ahorro_costes = data["ahorro_costes"]
predict_destino_alta = data["predict_destino_alta"]
predict_movilidad_alta = data["predict_movilidad_alta"]
predict_vivo_alta = data["predict_vivo_alta"]

# ____________________________________ VISUAL __________________________________________
# ______________________________________________________________________________________
# Obtener la fecha actual
import streamlit as st
from datetime import datetime
fecha_actual = datetime.now().strftime("%Y-%m-%d")

# Posicionar la fecha en la esquina superior derecha
st.markdown(
    f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; padding-right: 10px;">
        <p style="font-size: 16px; margin: 0;">Fecha de ingreso: {'2024-11-26'}</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown(
    f"""
    <div style="display: flex; justify-content: flex-end; align-items: center; padding-right: 10px;">
        <p style="font-size: 16px; margin: 0;">Fecha del documento: {fecha_actual}</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center;'>Predicción de Fractura de Cadera</h1>", unsafe_allow_html=True)
st.markdown("""---""")

# ----------------------------------- DATOS PACIENTE ------------------------------
st.header("Datos del paciente")
st.success(f"**ID paciente: {id_paciente}**")

tabla_resumen = pd.DataFrame({
    "Variable": ["Sexo", "Edad", "Lugar de residencia", "Tipo de fractura", "Lado de fractura"],
    "Valor": [sexo, edad, lugar_residencia, tipo_fractura, lado_fractura]
})

st.subheader("Resumen paciente")
st.table(tabla_resumen)

# _____________________________________________ RESUMEN PREDICCIONES ______________________________
#__________________________________________________________________________________________________

# ------------------------------------- DEMORA -------------------------------------
with st.container():
    st.header("Resumen predicciones")
    st.subheader("Demora")

    st.info(f"**Predicción de demora:** {resultado_demora} días")

# ------------------------------------- ESTANCIA -----------------------------------
    st.subheader("Estancia Hospitalaria")

    st.info(f"**Predicción de postoperatorio:** {postoperatorio_fijo} días")
    st.info(f"**Predicción de estancia total:** {postoperatorio_fijo + resultado_demora} días")

# ------------------------------------- DESTINO -----------------------------------
    st.subheader("Destino al alta")
    st.info(f"**Predicción de destino al alta:** Domicilio")
    
# ------------------------------------- MOVILIDAD -----------------------------------
    st.subheader("Movilidad al alta")
    st.info(f"**Predicción de movilidad al alta:** Muy limitada")

# ------------------------------------- VIVO 30 DIAS-----------------------------------
    st.subheader("Vivo a los 30 días")
    st.info(f"**Predicción de vivo a los 30 días:** Vivo a 30 días")

    st.markdown("<div class='no-overlap'></div>", unsafe_allow_html=True)

# _____________________________________________ GRAFICOS Y TABLAS __ ______________________________
#__________________________________________________________________________________________________
def crear_tabla_y_grafico(titulo, categorias, porcentajes, orden, colores=None):
    """
    Crea y muestra una tabla y un gráfico circular a partir de las categorías y porcentajes proporcionados.
    """
    # Crear DataFrame
    data = {"Categoría": categorias, "Porcentaje predicción": porcentajes}
    df = pd.DataFrame(data)

    # Formatear porcentaje en la tabla
    df_tabla = df.copy()
    df_tabla["Porcentaje predicción"] = df_tabla["Porcentaje predicción"].apply(lambda x: f"{x*100:.0f}%")

    # Mostrar tabla
    st.subheader(titulo)
    st.dataframe(df_tabla)

    # Crear gráfico circular
    fig = px.pie(
        df,
        values="Porcentaje predicción",
        names="Categoría",
        title=titulo,
        color="Categoría",
        category_orders={"Categoría": orden},
        color_discrete_sequence=colores,
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------- DESTINO ALTA ----------------------------------
with st.container():
    st.markdown("<div class='no-overlap'></div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Dos líneas vacías

    crear_tabla_y_grafico(
    titulo="Probabilidad de destino al alta",
    categorias=["Domicilio", "Fallecido", "Hospitalización Agudos", "Residencia/Institucionalizado"],
    porcentajes=predict_destino_alta,
    orden=["Residencia/Institucionalizado", "Domicilio", "Hospitalización Agudos", "Fallecido"],
)

# --------------------------------------- MOVILIDAD ALTA --------------------------------------
with st.container():
    crear_tabla_y_grafico(
    titulo="Probabilidad de movilidad al alta",
    categorias=["Completamente inmóvil", "Muy limitada", "Ligeramente limitada", "Sin limitaciones"],
    porcentajes=predict_movilidad_alta,
    orden=["Ligeramente limitada", "Sin limitaciones", "Muy limitada", "Completamente inmóvil"],
)

    # Separador visual
    st.markdown("<div class='no-overlap'></div>", unsafe_allow_html=True)

# --------------------------VIVO A LOS 30 DIAS ------------------------
with st.container():
    st.markdown("<div class='no-overlap'></div>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)  # Dos líneas vacías

    crear_tabla_y_grafico(
    titulo="Probabilidad de estar vivo a los 30 días",
    categorias=["Fallece", "Vivo a 30 días"],
    porcentajes=predict_vivo_alta,
    orden=["Fallece", "Vivo a 30 días"],
    colores=["#ff2b2b", "#83c9ff"],
)