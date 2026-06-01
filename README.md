# Road Safety Analysis — Community of Madrid (TFG)

Interactive dashboard to explore road safety indicators across the 
Community of Madrid road network (2016–2024).

## Structure
- `data/` — Computed indicators from Phase 4 (CSV)
- `code/indicators_phase4.py` — Phase 4: indicator calculation pipeline
- `code/app.py` — Phase 5: Streamlit interactive viewer

## How to run the viewer
pip install -r requirements.txt
streamlit run code/app.py

## Data sources
- Accident records: Portal de Datos Abiertos de la Comunidad de Madrid
- Traffic volume (IMD): DGT
- Road network geometry: CNIG / IDEM