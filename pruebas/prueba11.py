import asyncio
from pyppeteer import launch
from PIL import Image

async def capture_full_page(url, output_image_path, output_pdf_path):
    browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
    page = await browser.newPage()

    # Configurar el tamaño de la página
    await page.goto(url, {'waitUntil': 'networkidle2'})
    await page.setViewport({'width': 1920, 'height': 1080})
    
    # Capturar la página completa
    full_page = await page.screenshot({'path': output_image_path, 'fullPage': True})
    print(f"Captura completa guardada en {output_image_path}")
    
    # Convertir a PDF
    await page.pdf({'path': output_pdf_path, 'format': 'A4'})
    print(f"PDF guardado en {output_pdf_path}")
    
    await browser.close()

# Ejecutar captura
url = "http://99.81.70.181:8000/"
output_image_path = "./output/prueba11.png"
output_pdf_path = "./output/prueba11.pdf"

asyncio.run(capture_full_page(url, output_image_path, output_pdf_path))
