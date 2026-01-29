"""
Estilos CSS para la generación de PDF
Oculta elementos de Streamlit y prepara el diseño para impresión
"""

CSS_OCULTAR_STREAMLIT = """
@page { margin: 1in; }
body { margin: 0; padding: 1em; box-sizing: border-box; }
.no-overlap { page-break-before: always; }

/* --- 1. OCULTAR ESTRUCTURA DE STREAMLIT --- */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
header { display: none !important; }
footer { display: none !important; }

/* Ajustar contenido principal */
.main .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* --- 2. ZONA NUCLEAR: OCULTAR TODOS LOS BOTONES --- */

/* Ocultar etiquetas button estándar */
button { display: none !important; }

/* Ocultar contenedores de botones de Streamlit */
[data-testid="stButton"] { display: none !important; }

/* --- 3. LA SOLUCIÓN AL BOTÓN DE DESCARGA --- */

/* Opción A: Por ID de test de Streamlit */
[data-testid="stDownloadButton"] { display: none !important; }

/* Opción B: Por clase CSS específica */
.stDownloadButton { display: none !important; }

/* Opción C (LA DEFINITIVA): Ocultar cualquier enlace de descarga */
a[download] { display: none !important; visibility: hidden !important; }

/* --- 4. LIMPIEZA FINAL --- */
hr:last-of-type { display: none !important; }
.main .block-container > div:last-child { display: none !important; }
"""