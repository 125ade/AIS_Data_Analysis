"""
Script che per ogni anno nel dataset filtra i dati nell'area di interesse (poligono) e salva:
1) Un file CSV con i soli punti che hanno Bearing compreso tra 180° e 270°.
2) Una mappa interattiva (HTML) con gli stessi punti, colorati in base al tipo di nave (colonna "Type").
"""

import argparse
import os
import multiprocessing as mp
import pandas as pd
import numpy as np
import folium
import seaborn as sns
from shapely.geometry import Point, Polygon, shape
from tqdm import tqdm
from branca.element import Template, MacroElement

# Coordinate dell'antenna AIS
ais_antenna_coords = [43.58693627326397, 13.51656777958965]  # Lat, Lon

# Definizione del poligono di selezione (Area of Focus)
polygon_coords = [
    [13.627332698482832, 46.01607627362711],
    [11.827331194451205, 45.56839707274841],
    [12.206288202161488, 44.06768319786181],
    [13.437866762923107, 43.47479960302897],
    [14.06942488652328, 42.20912261569302],
    [16.364148350192465, 43.688312063316346],
    [14.522085391698596, 44.50473436137818],
    [13.627332698482832, 46.01607627362711]
]
area_polygon = Polygon(polygon_coords)

# GeoJSON con le geometrie del porto (per mostrare i poligoni nella mappa)
PORT_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "porto commerciale"},
            "geometry": {
                "coordinates": [
                    [
                        [13.504509281247067, 43.61303229602851],
                        [13.50444886853657, 43.61329028467097],
                        [13.5049624971183, 43.61392946547258],
                        [13.504946191446976, 43.61439240170344],
                        [13.504758676249935, 43.61484046298213],
                        [13.504298041094728, 43.61526544846015],
                        [13.504440715700298, 43.61575240729442],
                        [13.50582672524817, 43.61729562313022],
                        [13.50570443272889, 43.61741662130518],
                        [13.505965323436016, 43.617561228561215],
                        [13.505133734306867, 43.61843328430052],
                        [13.50550061186459, 43.61888775539188],
                        [13.507918410553884, 43.618566084649416],
                        [13.508215989017117, 43.61975537248219],
                        [13.50848503255969, 43.61988816991146],
                        [13.508608225374275, 43.62038987619593],
                        [13.506703519544743, 43.62061684667057],
                        [13.50674767840917, 43.62083693122386],
                        [13.506484423640273, 43.62086766922522],
                        [13.506498010982767, 43.62091316143949],
                        [13.507431371205627, 43.620780765815255],
                        [13.507505046117672, 43.62110839339081],
                        [13.509068006745451, 43.62092743652022],
                        [13.509233775297247, 43.62152744931541],
                        [13.509141208059816, 43.62218403387487],
                        [13.507669991905885, 43.622066120782875],
                        [13.507522870289904, 43.62239323395272],
                        [13.508715606243783, 43.6229637759196],
                        [13.508510686850144, 43.623233830563066],
                        [13.506760205902992, 43.62394203082408],
                        [13.506973813958155, 43.6248902280833],
                        [13.506226764728183, 43.62495413047563],
                        [13.505621456831989, 43.62462733374096],
                        [13.504527288756037, 43.623128093959224],
                        [13.50358943040385, 43.62349583547979],
                        [13.504624982333553, 43.62507992708237],
                        [13.504410056462177, 43.62516478795442],
                        [13.5008002817913, 43.62487157919165],
                        [13.497661084556142, 43.62572501974131],
                        [13.494026411994355, 43.62549293021135],
                        [13.49072313165675, 43.62820625240431],
                        [13.484487866532731, 43.621005251042675],
                        [13.489637349559615, 43.615665180034284],
                        [13.490945596598635, 43.61604806883244],
                        [13.493450750504138, 43.61945365665014],
                        [13.493283740243953, 43.62019923297919],
                        [13.495454873628177, 43.622395606805895],
                        [13.496178584755597, 43.619776069169916],
                        [13.498433223270837, 43.6200984799616],
                        [13.497987862576082, 43.622456056327024],
                        [13.49876724379115, 43.622234407785015],
                        [13.499296109615045, 43.620400738507925],
                        [13.50263631482241, 43.61764005400991],
                        [13.5016899233471, 43.6170153559907],
                        [13.50271981995246, 43.61600776486324],
                        [13.501355902825765, 43.61512107070553],
                        [13.501876311172708, 43.613102692454675],
                        [13.504509281247067, 43.61303229602851]
                    ]
                ],
                "type": "Polygon"
            },
            "id": 0
        },
        {
            "type": "Feature",
            "properties": {"name": "porto Marina Dorica"},
            "geometry": {
                "coordinates": [
                    [
                        [13.481774376708785, 43.60725294555081],
                        [13.49036836767985, 43.60691998305424],
                        [13.495961055305145, 43.60820943126123],
                        [13.50015666103917, 43.61046252809024],
                        [13.490072293087792, 43.615440113472914],
                        [13.48947288289125, 43.61566825442378],
                        [13.489207389334354, 43.61562150746752],
                        [13.488925394681871, 43.614489634272616],
                        [13.488168614442799, 43.61392642457611],
                        [13.485863577387647, 43.61210264232349],
                        [13.484312663433712, 43.61162415328829],
                        [13.481895256305194, 43.61140868842091],
                        [13.480332884385035, 43.61153471670434],
                        [13.479846085258743, 43.611037462191206],
                        [13.480781218824745, 43.60802950173462],
                        [13.48073026435597, 43.607944643479755],
                        [13.480714978015328, 43.6074748463389],
                        [13.480971448839227, 43.60733218470105],
                        [13.481774376708785, 43.60725294555081]
                    ]
                ],
                "type": "Polygon"
            },
            "id": 1
        },
        {
            "type": "Feature",
            "properties": {"name": "Fincantieri"},
            "geometry": {
                "coordinates": [
                    [
                        [13.505189782584836, 43.625339842782125],
                        [13.506093727335838, 43.627629973726215],
                        [13.502772712926173, 43.62842652055886],
                        [13.502812014871836, 43.62892435696901],
                        [13.50153470163798, 43.629251504365016],
                        [13.500473549104413, 43.6300907004115],
                        [13.499078330033683, 43.629251504365016],
                        [13.497683110961532, 43.62590882755987],
                        [13.500846917588603, 43.62509802262102],
                        [13.503224685301689, 43.62532561809374],
                        [13.505189782584836, 43.625339842782125]
                    ]
                ],
                "type": "Polygon"
            },
            "id": 2
        }
    ]
}

# MMSI da escludere in base al prefisso
excluded_prefixes = [
    "992471",  # AtoN fisici IT
    "992476",  # AtoN virtuali IT
    "992478",  # AtoN mobili IT
    "992381",  # AtoN fisici HR
    "992386",  # AtoN virtuali HR
    "992388",  # AtoN mobili HR
    "992781",  # AtoN fisici SI
    "992786",  # AtoN virtuali SI
    "992788",  # AtoN mobili SI
]

# MMSI da escludere in modo diretto
mmsi_exclude = [
    "0",
    "2470017",  # Italia
    "2470018",  # Italia
    "992467018",  # Italia
    "2470059",  # Italia
    "2470058",  # Italia
    "2470020",  # Italia
    "2780202",  # Slovenia
    "2386240",  # Croazia
    "2386300",  # Croazia
    "2386010",  # Croazia
    "2386020",  # Croazia
    "2386260",  # Croazia
    "2386190",  # Croazia
    "2386030",  # Croazia
    "2386080"  # Croazia
]


def create_unique_directory(base_path="results", prefix="analysis_bearing"):
    """Crea una directory univoca per salvare i risultati."""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    n = 1
    while True:
        dir_name = os.path.join(base_path, f"{prefix}_{n}")
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            return dir_name
        n += 1


def load_csv(file_path):
    """Carica un singolo file CSV."""
    return pd.read_csv(file_path)


def filter_by_polygon(chunk):
    """Tieni solo i record che cadono dentro il poligono definito."""
    return chunk[chunk.apply(lambda row: area_polygon.contains(Point(row['Longitude'], row['Latitude'])), axis=1)]


def filter_by_mmsi(chunk):
    """Escludi i record il cui MMSI ha un prefisso in excluded_prefixes o è in mmsi_exclude."""
    chunk = chunk[~chunk['MMSI'].astype(str).str.startswith(tuple(excluded_prefixes))]
    chunk = chunk[~chunk['MMSI'].astype(str).isin(mmsi_exclude)]
    return chunk


def generate_filtered_bearing_map(data, year_dir, year):
    """
  Filtra i dati per i punti con Bearing tra 180 e 270.
  Crea un file CSV e una mappa HTML con i soli punti filtrati.
  """
    # Filtro Bearing 180-270
    filtered_data = data[
        (data['Bearing'] >= 180) &
        (data['Bearing'] <= 270) &
        data['Bearing'].notna() &
        data['Longitude'].notna() &
        data['Latitude'].notna()
        ]
    if filtered_data.empty:
        print(f"[{year}] Nessun punto con Bearing 180-270.")
        return

    # Salvataggio CSV
    csv_path = os.path.join(year_dir, f"bearing_180_270_{year}.csv")
    filtered_data.to_csv(csv_path, index=False)
    print(f"[{year}] Salvataggio CSV -> {csv_path}")

    # Creazione mappa
    m = folium.Map(location=ais_antenna_coords, zoom_start=10)

    # Aggiunge poligono area di interesse
    folium.GeoJson(
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon_coords]
            },
        },
        style_function=lambda x: {"fill": False, "color": "red", "weight": 2},
    ).add_to(m)

    # Aggiunge poligoni del porto
    for feature in PORT_GEOJSON["features"]:
        geom = shape(feature["geometry"])
        folium.GeoJson(
            geom,
            style_function=lambda x: {
                "fillColor": "gray",
                "color": "black",
                "weight": 2,
                "fillOpacity": 0.4,
            },
        ).add_to(m)

    # Marker AIS
    folium.Marker(
        location=ais_antenna_coords,
        icon=folium.Icon(color='blue', icon='info-sign'),
        popup="Antenna AIS"
    ).add_to(m)

    # Differenzia i punti per "Type"
    colors = sns.color_palette("husl", len(filtered_data["Type"].unique())).as_hex()
    type_colors = {v_type: color for v_type, color in zip(filtered_data["Type"].unique(), colors)}

    # Legenda
    legend_html = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 180px; height: auto; 
        background-color: white; 
        border:2px solid grey; z-index:9999; font-size:14px;
        ">
    &emsp;<b>Bearing [180-270]</b><br>
    """
    for v_type, color in type_colors.items():
        legend_html += f'&emsp;<i style="background:{color};width:10px;height:10px;display:inline-block;"></i>&nbsp;{v_type}<br>'
    legend_html += """
    </div>
    {% endmacro %}
    """
    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    # Aggiunge i marker
    for _, row in filtered_data.iterrows():
        popup_text = f"""
        MMSI: {row['MMSI']}<br>
        Type: {row['Type']}<br>
        Bearing: {row['Bearing']}<br>
        Lat/Lon: {row['Latitude']}, {row['Longitude']}<br>
        Datetime: {row['datetime']}
        """
        folium.CircleMarker(
            location=(row["Latitude"], row["Longitude"]),
            radius=3,
            color=type_colors.get(row["Type"], "gray"),
            fill=True,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    # Salvataggio mappa HTML
    map_path = os.path.join(year_dir, f"map_bearing_180_270_{year}.html")
    m.save(map_path)
    print(f"[{year}] Salvataggio mappa HTML -> {map_path}")


def process_year_data(year, data, results_dir):
    """Per ogni anno, genera CSV + Mappa con punti bearing 180-270."""
    year_dir = os.path.join(results_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    generate_filtered_bearing_map(data, year_dir, year)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script per AIS Bearing 180-270")
    parser.add_argument(
        '-d', '--dataset',
        default='dataset/AIS_Dataset_csv',
        help="Cartella contenente i file CSV (default: dataset/AIS_Dataset_csv)"
    )
    args = parser.parse_args()

    dataset_folder = args.dataset
    csv_files = [
        os.path.join(dataset_folder, f)
        for f in os.listdir(dataset_folder)
        if f.endswith('.csv')
    ]
    if not csv_files:
        raise FileNotFoundError(f"Non ci sono file CSV nella cartella: {dataset_folder}")

    # Crea cartella risultati
    results_dir = create_unique_directory()

    # Caricamento parallelo dei CSV
    print("Caricamento parallelo dei CSV...")
    mp.freeze_support()
    mp.set_start_method('spawn', force=True)  # Per Windows
    with mp.Pool(mp.cpu_count()) as pool:
        data_chunks = list(tqdm(pool.imap(load_csv, csv_files), total=len(csv_files), desc="Loading CSV"))

    # Concatenazione
    data = pd.concat(data_chunks, ignore_index=True)
    del data_chunks

    # Conversione e creazione di colonne
    print("Preprocessing data...")
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
    data['year'] = data['datetime'].dt.year

    # Filtraggio area poligono, e MMSI da escludere
    print("Filtro area poligono...")
    num_proc = mp.cpu_count()
    data_splits = np.array_split(data.index, num_proc)
    data_parts = [data.loc[idx] for idx in data_splits]

    with mp.Pool(processes=num_proc) as pool:
        filtered_area = list(
            tqdm(pool.imap(filter_by_polygon, data_parts), total=num_proc, desc="Filtering by polygon")
        )
    data = pd.concat(filtered_area, ignore_index=True)

    print("Filtro MMSI indesiderati...")
    data_splits = np.array_split(data.index, num_proc)
    data_parts = [data.loc[idx] for idx in data_splits]
    with mp.Pool(processes=num_proc) as pool:
        filtered_mmsi = list(
            tqdm(pool.imap(filter_by_mmsi, data_parts), total=num_proc, desc="Filtering by MMSI")
        )
    data = pd.concat(filtered_mmsi, ignore_index=True)

    print(f"Totale righe dopo i filtri: {len(data)}")

    # Raggruppa e processa per anno
    years = sorted(data['year'].unique())
    for year in years:
        df_year = data[data['year'] == year]
        process_year_data(year, df_year, results_dir)

    print(f"Risultati salvati in: {results_dir}")
