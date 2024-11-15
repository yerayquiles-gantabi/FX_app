import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from fpdf import FPDF
import base64
from io import BytesIO
import time

# Configurar función para capturar la página
def capture_screenshot():
    # Configuración de Selenium WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # Iniciar WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get("http://localhost:8501")  # Cambia al URL local donde esté corriendo tu Streamlit
    time.sleep(3)  # Dar tiempo para que la página cargue completamente
    
    # Tomar captura de pantalla
    screenshot = driver.get_screenshot_as_png()
    driver.quit()
    
    return screenshot

# Convertir la captura de pantalla en un PDF y codificar en base64 para descarga
def convert_screenshot_to_pdf_and_download(screenshot):
    pdf = FPDF()
    pdf.add_page()
    
    # Guardar la imagen temporalmente para añadirla al PDF
    image_path = "/tmp/screenshot.png"
    with open(image_path, "wb") as f:
        f.write(screenshot)
    
    pdf.image(image_path, x=10, y=10, w=190)  # Ajusta según sea necesario
    pdf_buffer = BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    
    # Codificar PDF en base64
    pdf_base64 = base64.b64encode(pdf_buffer.read()).decode('utf-8')
    href = f'<a href="data:application/pdf;base64,{pdf_base64}" download="streamlit_visual.pdf">Descargar PDF</a>'
    st.markdown(href, unsafe_allow_html=True)

# Contenido de la aplicación Streamlit
st.title("Ejemplo de Streamlit")
st.write("Captura esta visualización como PDF.")

# Botón para capturar y descargar PDF
if st.button("Capturar visual y descargar PDF"):
    screenshot = capture_screenshot()
    convert_screenshot_to_pdf_and_download(screenshot)
