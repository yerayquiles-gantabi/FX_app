import streamlit as st
import pandas as pd
import plotly.express as px
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.utils import ChromeType
from webdriver_manager.chrome import ChromeDriverManager
import time
import os

# Función para guardar la página de Streamlit como PDF
def save_streamlit_as_pdf():
    options = Options()
    options.add_argument('--headless')  # Ejecuta en modo headless
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--kiosk-printing')  # Imprimir automáticamente

    # Usar Chromium en lugar de Chrome
    service = Service(ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install())
    driver = webdriver.Chrome(service=service, options=options)

    # URL de tu aplicación Streamlit
    driver.get("http://localhost:8502")  # Cambia si tu aplicación está en otra URL

    # Espera que la página cargue completamente
    time.sleep(5)

    # Guardar la página como PDF
    pdf_path = os.path.join(os.getcwd(), "streamlit_page.pdf")
    driver.execute_cdp_cmd("Page.printToPDF", {
        "path": pdf_path,
        "displayHeaderFooter": False,
        "printBackground": True,
        "preferCSSPageSize": True,
    })

    driver.quit()
    return pdf_path

# Interfaz Streamlit
st.title("Generar PDF Automáticamente con Chromium")
st.write("Esta página incluye un gráfico que será generado automáticamente en un PDF.")

# Datos y gráfico
df = pd.DataFrame({
    'Días': [1, 2, 3, 4, 5, 6, 7],
    'Pacientes': [5, 10, 8, 12, 15, 10, 7]
})
fig = px.bar(df, x='Días', y='Pacientes', title='Pacientes por Días en el Hospital')
st.plotly_chart(fig)

# Botón para generar PDF
if st.button("Generar PDF"):
    pdf_path = save_streamlit_as_pdf()

    # Descargar el PDF
    with open(pdf_path, "rb") as pdf_file:
        st.download_button(
            label="Descargar PDF",
            data=pdf_file,
            file_name="streamlit_reporte.pdf",
            mime="application/pdf"
        )

    # Limpieza del archivo PDF temporal
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
