from fpdf import FPDF
import streamlit as st
import pandas as pd
import plotly.express as px

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.set_text_color(30, 144, 255)  # Azul
        self.cell(0, 10, 'Proyecto Fractura Cadera', ln=True, align='C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 10)
        self.set_text_color(128)
        self.cell(0, 10, f'Página {self.page_no()}', align='C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_chart(self, image_path):
        self.image(image_path, x=10, w=190)
        self.ln(10)

# Contenido de la aplicación Streamlit
st.title("Proyecto Fractura Cadera")
st.subheader("Hospital San Juan de Dios-León")
st.markdown("---")

dias_hospital = st.number_input("Introduce los días en el hospital:", min_value=0, step=1)

# Crear un gráfico
df = pd.DataFrame({
    'Días': [1, 2, 3, 4, 5, 6, 7],
    'Pacientes': [5, 10, 8, 12, 15, 10, 7]
})
fig = px.bar(df, x='Días', y='Pacientes', title='Pacientes por Días en el Hospital')

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Guardar el gráfico como imagen
image_path = "chart.png"
fig.write_image(image_path)

if st.button("Generar PDF"):
    pdf = PDF()
    pdf.add_page()
    
    # Título de capítulo
    pdf.chapter_title('Informe de Hospitalización')
    
    # Cuerpo de texto
    body_text = f"Este informe incluye datos sobre días de hospitalización.\n\nDías en el hospital: {dias_hospital}"
    pdf.chapter_body(body_text)
    
    # Agregar gráfico
    pdf.chapter_title('Gráfico: Pacientes por Días')
    pdf.add_chart(image_path)
    
    # Guardar PDF
    pdf_file = "informe_avanzado.pdf"
    pdf.output(pdf_file)
    
    # Descargar el PDF
    with open(pdf_file, "rb") as file:
        st.download_button(label="Descargar PDF", data=file, file_name="informe_avanzado.pdf")

# Limpieza del archivo temporal del gráfico
import os
if os.path.exists(image_path):
    os.remove(image_path)
