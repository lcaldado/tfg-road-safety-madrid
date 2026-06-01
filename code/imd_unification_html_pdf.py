import zipfile

CARPETA_BASE = 'data/raw/imd'
CARPETA_PROCESADOS = 'data/processed'
os.makedirs(CARPETA_PROCESADOS, exist_ok=True)

# Configuración: qué años tienen qué formato
AÑOS_HTML = {
    2018: 'imd_trafico_2018.zip',
    2019: 'imd_trafico_2019_.zip',
    2020: 'imd_trafico_2020_.zip',
    2021: 'imd_trafico_2021_.zip',
    2022: 'imd_trafico_2022_.zip',
    2023: 'imd_trafico_2023.zip',
}
AÑOS_PDF = {
    2024: 'imd_trafico_2024.zip',
}

todos = []

# --- Procesar años HTML ---
for año, nombre_zip in AÑOS_HTML.items():
    ruta_zip = os.path.join(CARPETA_BASE, nombre_zip)
    carpeta_ext = os.path.join(CARPETA_BASE, str(año))
    
    # Extraer ZIP si no está ya extraído
    if not os.path.exists(carpeta_ext):
        with zipfile.ZipFile(ruta_zip, 'r') as z:
            z.extractall(carpeta_ext)
    
    df = procesar_zip_html(carpeta_ext, año)
    
    # Normaliza columnas para que coincidan con el formato PDF
    df_norm = df[['año', 'carretera', 'pk', 'imd_total', 'fichero']].copy()
    df_norm['formato_fuente'] = 'html_calculado'
    todos.append(df_norm)
    
    df.to_csv(f'{CARPETA_PROCESADOS}/imd_{año}_detalle.csv', index=False)

# --- Procesar años PDF ---
for año, nombre_zip in AÑOS_PDF.items():
    ruta_zip = os.path.join(CARPETA_BASE, nombre_zip)
    carpeta_ext = os.path.join(CARPETA_BASE, str(año))
    
    if not os.path.exists(carpeta_ext):
        with zipfile.ZipFile(ruta_zip, 'r') as z:
            z.extractall(carpeta_ext)
    
    df = procesar_zip_pdf(carpeta_ext, año)
    
    df_norm = df[['año', 'carretera', 'pk', 'imd_total', 'pct_pesados', 'fichero']].copy()
    df_norm['formato_fuente'] = 'pdf_oficial'
    todos.append(df_norm)
    
    df.to_csv(f'{CARPETA_PROCESADOS}/imd_{año}_detalle.csv', index=False)

# --- Dataset final unificado ---
df_final = pd.concat(todos, ignore_index=True)
df_final.to_csv(f'{CARPETA_PROCESADOS}/imd_todos_años.csv', index=False)

print(f"\nDataset final: {len(df_final)} filas")
print(df_final.groupby(['año', 'formato_fuente']).size())
print(df_final.head())