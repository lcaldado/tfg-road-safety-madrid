import re
import os
import pandas as pd
from bs4 import BeautifulSoup

def extraer_imd_desde_html(ruta_html):
    """
    Extrae los datos de intensidad horaria de un archivo HTML de la CM.
    Calcula la IMD como suma de las 24 horas de cada día.
    Devuelve un dict con metadatos e IMD calculada.
    """
    with open(ruta_html, 'r', encoding='utf-8') as f:
        contenido = f.read()

    soup = BeautifulSoup(contenido, 'html.parser')

    # --- Extraer metadatos del título ---
    # Ejemplo: "Est. 111 - M-100 PK 0,150 Sent.: Ambos"
    titulo = soup.find('title')
    titulo_text = titulo.text.strip() if titulo else ''
    
    # Extraer carretera y PK del título
    match_carretera = re.search(r'(M-\d+)\s+PK\s+([\d,]+)', titulo_text)
    carretera = match_carretera.group(1) if match_carretera else 'DESCONOCIDA'
    pk = match_carretera.group(2).replace(',', '.') if match_carretera else None

    # Extraer número de estación
    match_est = re.search(r'Est\.\s*(\d+)', titulo_text)
    estacion = match_est.group(1) if match_est else None

    # --- Extraer arrays de datos del JavaScript ---
    # Busca patrones como: y: [32,15,14,...],
    # El primer bloque (data) es calzada total
    patron_y = r'y:\s*\[([^\]]+)\]'
    arrays_y = re.findall(patron_y, contenido)

    dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
    
    resultados = {}

    # Los primeros 7 arrays corresponden a calzada TOTAL (los que necesitas para IMD)
    for i, dia in enumerate(dias):
        if i < len(arrays_y):
            valores = [int(v.strip()) for v in arrays_y[i].split(',') if v.strip().isdigit()]
            # IMD de ese día = suma de las 24 horas
            imd_dia = sum(valores)
            resultados[f'imd_{dia.lower()}'] = imd_dia

    # IMD media diaria ponderada (5 laborables, 1 sábado, 1 domingo)
    imd_laborable = sum([
        resultados.get('imd_lunes', 0),
        resultados.get('imd_martes', 0),
        resultados.get('imd_miercoles', 0),
        resultados.get('imd_jueves', 0),
        resultados.get('imd_viernes', 0)
    ]) / 5

    imd_sabado = resultados.get('imd_sabado', 0)
    imd_domingo = resultados.get('imd_domingo', 0)

    # Fórmula estándar de la DGT: IMD = (5*L + S + D) / 7
    imd_total = (5 * imd_laborable + imd_sabado + imd_domingo) / 7

    return {
        'estacion': estacion,
        'carretera': carretera,
        'pk': pk,
        'imd_laborable_media': round(imd_laborable),
        'imd_sabado': imd_sabado,
        'imd_domingo': imd_domingo,
        'imd_total': round(imd_total),
        **resultados  # incluye imd por día si quieres el detalle
    }


def procesar_zip_html(carpeta_extraida, año):
    """
    Procesa todos los HTMLs de una carpeta (año extraído del ZIP).
    Solo coge los archivos GLineas_*.html (calzada total, no bimestral).
    """
    registros = []

    # Solo los archivos de gráficas de líneas (no los Bimestral)
    htmls = [f for f in os.listdir(carpeta_extraida) 
             if f.startswith('GLineas_') and f.endswith('.html')]
    
    print(f"Año {año}: {len(htmls)} archivos GLineas encontrados")

    for nombre in sorted(htmls):
        ruta = os.path.join(carpeta_extraida, nombre)
        try:
            datos = extraer_imd_desde_html(ruta)
            datos['año'] = año
            datos['fichero'] = nombre
            registros.append(datos)
        except Exception as e:
            print(f"  ERROR en {nombre}: {e}")

    return pd.DataFrame(registros)