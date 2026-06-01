import pdfplumber
import re
import os

def extraer_imd_desde_pdf(ruta_pdf):
    """
    Extrae la IMD anual total de un PDF bimestral de la CM (formato 2024).
    Busca la fila 'Media Diaria / 00:00 a 24:00' columna Total.
    """
    with pdfplumber.open(ruta_pdf) as pdf:
        pagina = pdf.pages[0]  # Siempre está en la primera página
        tabla = pagina.extract_table()

    if not tabla:
        raise ValueError(f"No se encontró tabla en {ruta_pdf}")

    # --- Extraer metadatos del nombre del archivo ---
    nombre = os.path.basename(ruta_pdf)
    # Expresión regular flexible: soporta espacios o guiones bajos, y comas
    match = re.search(r'Bimestral_([A-Za-z0-9\-]+)[\s_]+PK[\s_]+([\d_,]+)\.pdf', nombre, re.IGNORECASE)
    carretera = match.group(1) if match else 'DESCONOCIDA'
    # Reemplazamos tanto guiones bajos como comas por el punto decimal
    pk = match.group(2).replace('_', '.').replace(',', '.') if match else None

    # --- Buscar la fila objetivo ---
    # Necesitamos: tipo_dato="Media Diaria", horas="00:00 a 24:00"
    # En la tabla extraída, las columnas son:
    # [tipo_dato, horas, IMD_EneFeb, Pes_EneFeb, IMD_MarAbr, Pes_MarAbr,
    #  IMD_MayJun, Pes_MayJun, IMD_JulAgo, Pes_JulAgo, IMD_SepOct, Pes_SepOct,
    #  IMD_NovDic, Pes_NovDic, IMD_Total, Pes_Total]

    imd_total = None
    pct_pesados_total = None
    tipo_dato_actual = ''

    for fila in tabla:
        if fila is None:
            continue
        
        # La columna tipo_dato a veces está vacía (celdas fusionadas)
        # pdfplumber rellena las celdas fusionadas con None
        col0 = str(fila[0]).strip() if fila[0] else ''
        col1 = str(fila[1]).strip() if fila[1] else ''
        
        if col0:
            tipo_dato_actual = col0  # Actualiza cuando hay valor nuevo
        
        # Detecta la fila que nos interesa
        es_media_diaria = 'Media' in tipo_dato_actual and 'Diaria' in tipo_dato_actual
        es_24h = '00:00' in col1 and '24:00' in col1
        
        if es_media_diaria and es_24h:
            # Las últimas dos columnas son IMD Total y % Pesados Total
            try:
                imd_total = float(str(fila[-2]).replace('.', '').replace(',', '.').strip())
                pct_pesados_total = float(str(fila[-1]).replace(',', '.').strip())
            except (ValueError, TypeError, IndexError):
                pass
            break

    return {
        'carretera': carretera,
        'pk': pk,
        'imd_total': imd_total,
        'pct_pesados': pct_pesados_total,
    }


def procesar_zip_pdf(carpeta_extraida, año):
    """
    Procesa todos los PDFs Bimestral_*.pdf de una carpeta.
    """
    registros = []

    pdfs = [f for f in os.listdir(carpeta_extraida)
            if f.startswith('Bimestral_') and f.endswith('.pdf')]
    
    print(f"Año {año}: {len(pdfs)} archivos PDF encontrados")

    for nombre in sorted(pdfs):
        ruta = os.path.join(carpeta_extraida, nombre)
        try:
            datos = extraer_imd_desde_pdf(ruta)
            datos['año'] = año
            datos['fichero'] = nombre
            registros.append(datos)
        except Exception as e:
            print(f"  ERROR en {nombre}: {e}")

    return pd.DataFrame(registros)