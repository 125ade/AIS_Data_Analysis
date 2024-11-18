"""
Script per generare grafici, heatmap e mappe interattive giornaliere.
Le mappe includono geometrie dettagliate del porto, con colorazione per tipo di nave e popup con informazioni sui punti.
Inoltre, è stata aggiunta una legenda dei colori dei tipi di nave.
"""

import os
import multiprocessing as mp
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend non interattivo per prevenire errori legati a Tcl/Tk
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import shape
from tqdm import tqdm
import folium
import json
from branca.element import Template, MacroElement

ais_antenna_coords = [43.58693627326397, 13.51656777958965]  # Lat, Lon dell'antenna AIS
# GeoJSON del porto
PORT_GEOJSON = {
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "porto commerciale"
      },
      "geometry": {
        "coordinates": [
          [
            [
              13.504509281247067,
              43.61303229602851
            ],
            [
              13.50444886853657,
              43.61329028467097
            ],
            [
              13.5049624971183,
              43.61392946547258
            ],
            [
              13.504946191446976,
              43.61439240170344
            ],
            [
              13.504758676249935,
              43.61484046298213
            ],
            [
              13.504298041094728,
              43.61526544846015
            ],
            [
              13.504440715700298,
              43.61575240729442
            ],
            [
              13.50582672524817,
              43.61729562313022
            ],
            [
              13.50570443272889,
              43.61741662130518
            ],
            [
              13.505965323436016,
              43.617561228561215
            ],
            [
              13.505133734306867,
              43.61843328430052
            ],
            [
              13.50550061186459,
              43.61888775539188
            ],
            [
              13.507918410553884,
              43.618566084649416
            ],
            [
              13.508215989017117,
              43.61975537248219
            ],
            [
              13.50848503255969,
              43.61988816991146
            ],
            [
              13.508608225374275,
              43.62038987619593
            ],
            [
              13.506703519544743,
              43.62061684667057
            ],
            [
              13.50674767840917,
              43.62083693122386
            ],
            [
              13.506484423640273,
              43.62086766922522
            ],
            [
              13.506498010982767,
              43.62091316143949
            ],
            [
              13.507431371205627,
              43.620780765815255
            ],
            [
              13.507505046117672,
              43.62110839339081
            ],
            [
              13.509068006745451,
              43.62092743652022
            ],
            [
              13.509233775297247,
              43.62152744931541
            ],
            [
              13.509141208059816,
              43.62218403387487
            ],
            [
              13.507669991905885,
              43.622066120782875
            ],
            [
              13.507522870289904,
              43.62239323395272
            ],
            [
              13.508715606243783,
              43.6229637759196
            ],
            [
              13.508510686850144,
              43.623233830563066
            ],
            [
              13.506760205902992,
              43.62394203082408
            ],
            [
              13.506973813958155,
              43.6248902280833
            ],
            [
              13.506226764728183,
              43.62495413047563
            ],
            [
              13.505621456831989,
              43.62462733374096
            ],
            [
              13.504527288756037,
              43.623128093959224
            ],
            [
              13.50358943040385,
              43.62349583547979
            ],
            [
              13.504624982333553,
              43.62507992708237
            ],
            [
              13.504410056462177,
              43.62516478795442
            ],
            [
              13.5008002817913,
              43.62487157919165
            ],
            [
              13.497661084556142,
              43.62572501974131
            ],
            [
              13.494026411994355,
              43.62549293021135
            ],
            [
              13.49072313165675,
              43.62820625240431
            ],
            [
              13.484487866532731,
              43.621005251042675
            ],
            [
              13.489637349559615,
              43.615665180034284
            ],
            [
              13.490945596598635,
              43.61604806883244
            ],
            [
              13.493450750504138,
              43.61945365665014
            ],
            [
              13.493283740243953,
              43.62019923297919
            ],
            [
              13.495454873628177,
              43.622395606805895
            ],
            [
              13.496178584755597,
              43.619776069169916
            ],
            [
              13.498433223270837,
              43.6200984799616
            ],
            [
              13.497987862576082,
              43.622456056327024
            ],
            [
              13.49876724379115,
              43.622234407785015
            ],
            [
              13.499296109615045,
              43.620400738507925
            ],
            [
              13.50263631482241,
              43.61764005400991
            ],
            [
              13.5016899233471,
              43.6170153559907
            ],
            [
              13.50271981995246,
              43.61600776486324
            ],
            [
              13.501355902825765,
              43.61512107070553
            ],
            [
              13.501876311172708,
              43.613102692454675
            ],
            [
              13.504509281247067,
              43.61303229602851
            ]
          ]
        ],
        "type": "Polygon"
      },
      "id": 0
    },
    {
      "type": "Feature",
      "properties": {
        "name": "porto Marina Dorica"
      },
      "geometry": {
        "coordinates": [
          [
            [
              13.481774376708785,
              43.60725294555081
            ],
            [
              13.49036836767985,
              43.60691998305424
            ],
            [
              13.495961055305145,
              43.60820943126123
            ],
            [
              13.50015666103917,
              43.61046252809024
            ],
            [
              13.490072293087792,
              43.615440113472914
            ],
            [
              13.48947288289125,
              43.61566825442378
            ],
            [
              13.489207389334354,
              43.61562150746752
            ],
            [
              13.488925394681871,
              43.614489634272616
            ],
            [
              13.488168614442799,
              43.61392642457611
            ],
            [
              13.485863577387647,
              43.61210264232349
            ],
            [
              13.484312663433712,
              43.61162415328829
            ],
            [
              13.481895256305194,
              43.61140868842091
            ],
            [
              13.480332884385035,
              43.61153471670434
            ],
            [
              13.479846085258743,
              43.611037462191206
            ],
            [
              13.480781218824745,
              43.60802950173462
            ],
            [
              13.48073026435597,
              43.607944643479755
            ],
            [
              13.480714978015328,
              43.6074748463389
            ],
            [
              13.480971448839227,
              43.60733218470105
            ],
            [
              13.481774376708785,
              43.60725294555081
            ]
          ]
        ],
        "type": "Polygon"
      },
      "id": 1
    },
    {
      "type": "Feature",
      "properties": {
        "name": "Fincantieri"
      },
      "geometry": {
        "coordinates": [
          [
            [
              13.505189782584836,
              43.625339842782125
            ],
            [
              13.506093727335838,
              43.627629973726215
            ],
            [
              13.502772712926173,
              43.62842652055886
            ],
            [
              13.502812014871836,
              43.62892435696901
            ],
            [
              13.50153470163798,
              43.629251504365016
            ],
            [
              13.500473549104413,
              43.6300907004115
            ],
            [
              13.499078330033683,
              43.629251504365016
            ],
            [
              13.497683110961532,
              43.62590882755987
            ],
            [
              13.500846917588603,
              43.62509802262102
            ],
            [
              13.503224685301689,
              43.62532561809374
            ],
            [
              13.505189782584836,
              43.625339842782125
            ]
          ]
        ],
        "type": "Polygon"
      },
      "id": 2
    }
  ]
}


def create_unique_directory(
        base_path="results",
        prefix=f"analysis_{os.path.splitext(os.path.basename(sys.argv[0]))[0].split('_')[-1]}"
):
  """Crea una directory unica per salvare i risultati."""
  if not os.path.exists(base_path):
    os.makedirs(base_path)

  n = 1
  while os.path.exists(os.path.join(base_path, f"{prefix}_{n}")):
    n += 1

  unique_directory = os.path.join(base_path, f"{prefix}_{n}")
  os.makedirs(unique_directory)
  return unique_directory


def load_csv(file):
  """Carica un singolo file CSV."""
  return pd.read_csv(file)


def process_year_data(year_data_tuple):
  """Processa i dati per un anno specifico: genera grafici, heatmap e mappe interattive."""
  year, data, results_dir = year_data_tuple
  year_dir = os.path.join(results_dir, str(year))
  os.makedirs(year_dir, exist_ok=True)

  # Genera grafici per l'anno
  generate_yearly_plots(data, year_dir, year)

  # Genera mappe interattive giornaliere
  generate_daily_maps_for_year(data, year_dir, year)


def generate_yearly_plots(data, year_dir, year):
  """Genera i grafici per un anno specifico."""
  print(f"Generating plots for year {year}...")

  plot_tasks = [
    ("Vessel Type Distribution", plot_vessel_type_distribution),
    ("Distance Distribution", plot_distance_distribution),
    ("Bearing Distribution", plot_bearing_distribution),
    ("Daily Messages", plot_daily_messages),
    ("Hourly Messages", plot_hourly_messages),
  ]

  for plot_name, plot_func in plot_tasks:
    print(f" - {plot_name}")
    plot_func(data, year_dir, year)


def plot_vessel_type_distribution(data, results_dir, year):
  """Plot della distribuzione dei tipi di nave."""
  type_counts = data['Type'].value_counts()
  plt.figure(figsize=(10, 6))
  sns.countplot(x='Type', data=data, order=type_counts.index)
  plt.title(f'Distribution of Vessel Types ({year})')
  plt.xlabel('Vessel Type')
  plt.ylabel('Count')
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, f'vessel_type_distribution_{year}.png'))
  plt.close()


def plot_distance_distribution(data, results_dir, year):
  """Plot della distribuzione delle distanze."""
  if 'Distance' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Distance'], bins=50, kde=True)
    plt.title(f'Distribution of Distances ({year})')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'distance_distribution_{year}.png'))
    plt.close()
  else:
    print(f"Distance column not found for year {year}. Skipping distance distribution plot.")


def plot_bearing_distribution(data, results_dir, year):
  """Plot della distribuzione delle direzioni."""
  if 'Bearing' in data.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Bearing'], bins=36, kde=True)
    plt.title(f'Distribution of Bearings ({year})')
    plt.xlabel('Bearing')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'bearing_distribution_{year}.png'))
    plt.close()
  else:
    print(f"Bearing column not found for year {year}. Skipping bearing distribution plot.")


def plot_daily_messages(data, results_dir, year):
  """Plot del numero di messaggi per giorno."""
  daily_counts = data.groupby('date').size()
  plt.figure(figsize=(12, 6))
  daily_counts.plot()
  plt.title(f'Number of AIS Messages per Day ({year})')
  plt.xlabel('Date')
  plt.ylabel('Number of Messages')
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, f'daily_messages_{year}.png'))
  plt.close()


def plot_hourly_messages(data, results_dir, year):
  """Plot del numero di messaggi per ora."""
  hourly_counts = data.groupby('hour').size()
  plt.figure(figsize=(10, 6))
  hourly_counts.plot(kind='bar')
  plt.title(f'Number of AIS Messages per Hour ({year})')
  plt.xlabel('Hour of the Day')
  plt.ylabel('Number of Messages')
  plt.tight_layout()
  plt.savefig(os.path.join(results_dir, f'hourly_messages_{year}.png'))
  plt.close()


def generate_daily_maps_for_year(data, year_dir, year):
  """Genera mappe interattive giornaliere per ogni anno specifico."""
  dates = sorted(data['date'].unique())

  print(f"Generating daily maps for year {year}...")
  for date in tqdm(dates, desc=f"Year {year}", unit="day"):
    day_data = data[data['date'] == date]

    # Controlla se ci sono abbastanza dati per generare la mappa
    day_data = day_data.dropna(subset=['Longitude', 'Latitude'])
    if day_data.empty:
      continue

    # Crea la mappa
    m = folium.Map(location=ais_antenna_coords, zoom_start=13)

    # Aggiungi poligoni del porto
    for feature in PORT_GEOJSON['features']:
      geom = shape(feature['geometry'])
      name = feature['properties']['name']
      folium.GeoJson(
        geom,
        name=name,
        style_function=lambda x: {
          "fillColor": "gray",
          "color": "black",
          "weight": 2,
          "fillOpacity": 0.4,
        },
      ).add_to(m)

    # Aggiungi marker per l'antenna AIS
    folium.Marker(
      location=ais_antenna_coords,
      icon=folium.Icon(color='blue', icon='info-sign'),
      popup="Antenna AIS"
    ).add_to(m)

    # Aggiungi punti delle navi con colorazione basata sul tipo
    colors = sns.color_palette("husl", len(day_data['Type'].unique())).as_hex()
    type_colors = {v_type: color for v_type, color in zip(day_data['Type'].unique(), colors)}

    # Aggiungi legenda dei tipi di nave
    legend_html = '''
      {% macro html(this, kwargs) %}
      <div style="
      position: fixed; 
      bottom: 50px; left: 50px; width: 150px; height: auto; 
      background-color: white; 
      border:2px solid grey; z-index:9999; font-size:14px;
      ">
      &emsp;<b>Vessel Types</b><br>
      '''
    for v_type, color in type_colors.items():
      legend_html += f'&emsp;<i style="background:{color};width:10px;height:10px;display:inline-block;"></i>&nbsp;{v_type}<br>'

    legend_html += '''
      </div>
      {% endmacro %}
      '''

    legend = MacroElement()
    legend._template = Template(legend_html)
    m.get_root().add_child(legend)

    # Aggiungi marker per le navi
    for _, row in day_data.iterrows():
      # Seleziona le colonne rilevanti per il popup
      popup_info = {
        'MMSI': row['MMSI'],
        'Type': row['Type'],
        'Latitude': row['Latitude'],
        'Longitude': row['Longitude'],
        'Datetime': row['datetime']
      }
      popup_text = "<br>".join([f"{k}: {v}" for k, v in popup_info.items()])

      folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=3,
        color=type_colors.get(row['Type'], 'gray'),
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_text, max_width=300)
      ).add_to(m)

    # Definisci la cartella delle mappe
    maps_dir = os.path.join(year_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)

    # Percorso completo per salvare la mappa
    map_path = os.path.join(maps_dir, f"map_{date}.html")

    # Salva la mappa come file HTML
    m.save(map_path)


if __name__ == '__main__':
  mp.freeze_support()  # Necessario per Windows
  mp.set_start_method('spawn')  # Compatibilità con Windows

  results_dir = create_unique_directory()

  dataset_folder = r'dataset/AIS_Dataset_csv'  # Assicurati che il percorso sia corretto
  csv_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]
  csv_files = csv_files[:1]  # Commenta o rimuovi questa linea per elaborare tutti i file
  if not csv_files:
    raise FileNotFoundError(f"No CSV files found in the folder {dataset_folder}. Please check the path.")

  print("Loading CSV files using parallel processing...")
  with mp.Pool(mp.cpu_count()) as pool:
    data_chunks = list(tqdm(pool.imap(load_csv, csv_files), total=len(csv_files), desc="Loading CSV files"))

  print("Concatenating dataframes...")
  data = pd.concat(data_chunks, ignore_index=True)
  del data_chunks  # Libera memoria

  print("Preprocessing data...")
  data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
  data['date'] = data['datetime'].dt.date
  data['year'] = data['datetime'].dt.year
  data['hour'] = data['datetime'].dt.hour
  data['Type'] = data['Type'].astype('category')

  # Ottieni la lista degli anni presenti nei dati
  years = sorted(data['year'].unique())

  # Prepara i dati per ogni anno
  year_data_list = []
  for year in years:
    year_data = data[data['year'] == year]
    year_data_list.append((year, year_data, results_dir))

  print("Processing data for each year in parallel...")
  with mp.Pool(processes=min(len(years), mp.cpu_count())) as pool:
    pool.map(process_year_data, year_data_list)

  print(f"All results have been saved in the folder '{results_dir}'.")
