import asyncio
from pyppeteer import launch
from PIL import Image

async def capture_full_page(url, output_image_path, output_pdf_path):
    # Iniciar el navegador
    browser = await launch(headless=True, args=['--no-sandbox', '--disable-setuid-sandbox'])
    page = await browser.newPage()

    # Configurar el tamaño inicial de la ventana
    await page.setViewport({'width': 1920, 'height': 1080})

    # Cargar la página y esperar hasta que esté completamente cargada
    await page.goto(url, {'waitUntil': 'networkidle2'})
    await asyncio.sleep(5)  # Asegurarse de que el contenido dinámico termine de cargar

    # Obtener el tamaño total de la página
    dimensions = await page.evaluate("""() => {
        return {
            width: document.body.scrollWidth,
            height: document.body.scrollHeight
        };
    }""")
    print(f"Tamaño de la página: {dimensions['width']}x{dimensions['height']}")

    # Ajustar el viewport al tamaño total de la página
    await page.setViewport({'width': dimensions['width'], 'height': dimensions['height']})

    # Capturar la página completa como imagen
    await page.screenshot({'path': output_image_path, 'fullPage': True})
    print(f"Captura completa guardada en {output_image_path}")

    # Guardar la página como PDF
    await page.pdf({'path': output_pdf_path, 'format': 'A4'})
    print(f"PDF guardado en {output_pdf_path}")

    await browser.close()

# Ejecutar la captura
url = "http://99.81.70.181:8000/"
output_image_path = "./output/prueba13.png"
output_pdf_path = "./output/prueba13.pdf"

asyncio.run(capture_full_page(url, output_image_path, output_pdf_path))
