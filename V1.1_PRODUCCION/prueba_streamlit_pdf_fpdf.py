import pickle
from pathlib import Path
import pandas as pd  
import plotly.express as px  
import plotly.figure_factory as ff
import streamlit as st 
import streamlit_authenticator as stauth
import numpy as np
import time
from catboost import CatBoostRegressor
from sklearn import preprocessing
import datetime
from catboost import CatBoostClassifier
from fpdf import FPDF
from io import BytesIO
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Función para crear el PDF con gráficos
def create_pdf(content, image_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', size=12)

    # Agregar contenido al PDF
    for line in content:
        pdf.cell(200, 10, txt=line, ln=True, align='L')
    
    # Agregar el gráfico al PDF
    pdf.image(image_path, x=10, y=pdf.get_y() + 10, w=180)
    
    # Guardar el PDF en la memoria
    pdf_output = 'report_with_graph.pdf'
    pdf.output(pdf_output)
    return pdf_output

st.markdown("<h1 style='text-align: center;'>Proyecto Fractura Cadera</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Modelos Predictivos y Simulador de Escenarios con Inteligencia Artificial para la Mejora del Proceso de Gestión de Pacientes Hospital San Juan de Dios-León</h2>", unsafe_allow_html=True)
st.markdown("""---""")
    
st.markdown("<span style='color:gray;font-size:90%'>Nota uso: Las predicciones se obtienen de las variables que aparecen en el lateral izquierdo. Las simulaciones surgen de los cambios que realizas en las variables de Demora y Postoperatorio</span>", unsafe_allow_html=True)
st.header('Días en el hospital')

# Ejemplo de contenido para el PDF
content = [
    "Proyecto Fractura Cadera",
    "Hospital San Juan de Dios-León",
    "Este informe incluye datos sobre días de hospitalización y simulaciones de escenarios."
]

dias_hospital = st.number_input("Introduce los días en el hospital:", min_value=0, step=1)
if dias_hospital:
    content.append(f"Días en el hospital: {dias_hospital}")

# Crear un gráfico con Plotly
df = pd.DataFrame({
    'Días': [1, 2, 3, 4, 5, 6, 7],
    'Pacientes': [5, 10, 8, 12, 15, 10, 7]
})
fig = px.bar(df, x='Días', y='Pacientes', title='Pacientes por Días en el Hospital')

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Guardar el gráfico como imagen
image_path = 'chart.png'
fig.write_image(image_path)

# Botón para generar y descargar el PDF
if st.button("Generar PDF"):
    pdf_file = create_pdf(content, image_path)
    with open(pdf_file, "rb") as file:
        st.download_button(label="Descargar PDF", data=file, file_name="informe_con_grafico.pdf")
