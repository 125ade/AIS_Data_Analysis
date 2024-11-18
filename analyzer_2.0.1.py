"""
Script ottimizzato per generare grafici e heatmap giornaliere per ogni anno in parallelo,
includendo la geometria del porto di Ancona e un'icona per l'antenna AIS.
Se il caricamento della mappa fallisce, visualizza i contorni della costa e del porto.
"""

import os
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')  # Usa backend non interattivo per l'esecuzione dello script

import pyproj
os.environ['PROJ_LIB'] = pyproj.datadir.get_data_dir()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as ctx
from scipy.stats import gaussian_kde
from pyproj import Transformer
from tqdm import tqdm


def create_unique_directory(base_path="results", prefix="analysis"):
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
    """Processa i dati per un anno specifico: genera grafici e heatmap giornaliere."""
    year, data, results_dir = year_data_tuple
    year_dir = os.path.join(results_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    # Generazione dei grafici per l'anno
    generate_yearly_plots(data, year_dir, year)

    # Generazione delle heatmap giornaliere
    calculate_daily_heatmaps_for_year(data, year_dir, year)


def generate_yearly_plots(data, year_dir, year):
    """Genera i grafici per un anno specifico."""
    print(f"Generating plots for year {year}...")

    # Barra di avanzamento per i grafici
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


def calculate_daily_heatmaps_for_year(data, year_dir, year):
    """Genera heatmap per ogni giorno di un anno specifico."""
    dates = sorted(data['date'].unique())
    total_days = len(dates)

    # Coordinate del porto di Ancona e dell'antenna AIS
    ancona_port_polygon = Polygon([
        (13.5040, 43.6190), (13.5120, 43.6190),
        (13.5120, 43.6230), (13.5040, 43.6230),
        (13.5040, 43.6190)
    ])
    ancona_port_polygon = gpd.GeoSeries([ancona_port_polygon], crs="EPSG:4326").to_crs("EPSG:3857")
    ancona_port_polygon_x, ancona_port_polygon_y = ancona_port_polygon[0].exterior.xy

    ais_antenna_lon, ais_antenna_lat = 13.5167, 43.6167  # Antenna AIS di Ancona

    print(f"Generating daily heatmaps for year {year}...")
    for date in tqdm(dates, desc=f"Year {year}", unit="day"):
        day_data = data[data['date'] == date]

        day_data = day_data.dropna(subset=['Longitude', 'Latitude'])
        day_data = day_data[(day_data['Latitude'] >= -90) & (day_data['Latitude'] <= 90)]
        day_data = day_data[(day_data['Longitude'] >= -180) & (day_data['Longitude'] <= 180)]

        if day_data.empty:
            continue

        # Controlla se ci sono abbastanza punti unici per calcolare la densità
        unique_points = day_data[['Longitude', 'Latitude']].drop_duplicates()
        if unique_points.shape[0] < 2:
            print(f"Not enough unique points to calculate KDE for date {date}. Skipping.")
            continue

        # Creazione della geometria
        geometry = [Point(xy) for xy in zip(day_data['Longitude'], day_data['Latitude'])]
        crs_wgs84 = 'EPSG:4326'
        crs_web_mercator = 'EPSG:3857'

        geo_data = gpd.GeoDataFrame(day_data, geometry=geometry, crs=crs_wgs84)
        geo_data = geo_data.to_crs(crs_web_mercator)

        # Converti le coordinate dell'antenna AIS in EPSG:3857
        transformer = Transformer.from_crs(crs_wgs84, crs_web_mercator, always_xy=True)
        ais_antenna_x, ais_antenna_y = transformer.transform(ais_antenna_lon, ais_antenna_lat)

        # Definizione dell'area di Ancona
        buffer_size = 10000  # in metri
        x_min, x_max = geo_data.total_bounds[0] - buffer_size, geo_data.total_bounds[2] + buffer_size
        y_min, y_max = geo_data.total_bounds[1] - buffer_size, geo_data.total_bounds[3] + buffer_size

        geo_data_ancona = geo_data.cx[x_min:x_max, y_min:y_max]
        if geo_data_ancona.empty:
            continue

        x = geo_data_ancona.geometry.x
        y = geo_data_ancona.geometry.y
        xy_coords = np.vstack([x, y])

        # Aggiungi un blocco try-except per gestire l'eccezione
        try:
            kde = gaussian_kde(xy_coords)
            z = kde(xy_coords)
        except np.linalg.LinAlgError as e:
            print(f"LinAlgError for date {date}: {e}. Skipping this date.")
            continue

        # Creazione della figura
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(x, y, c=z, s=10, cmap='viridis')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])

        # Aggiungi la mappa di base con fallback per eventuali errori
        # map_loaded = False
        # try:
        #     ctx.add_basemap(ax, source=ctx.providers['OpenStreetMap.Mapnik'], crs=crs_web_mercator)
        #     map_loaded = True
        # except Exception as e:
        #     print(f"Failed to add OpenStreetMap basemap for {date}: {e}. Retrying with Stamen.TonerLite...")
        #     try:
        #         ctx.add_basemap(ax, source=ctx.providers['Stamen.TonerLite'], crs=crs_web_mercator)
        #         map_loaded = True
        #     except Exception as e2:
        #         print(f"Failed to add Stamen basemap for {date}: {e2}. Using fallback contours.")

        #if not map_loaded:
        #    # Disegna il contorno del porto e della costa
        #    ax.plot(ancona_port_polygon_x, ancona_port_polygon_y, color='black', linestyle='--', label='Porto di Ancona (contorno)')
        # Disegna il poligono del porto di Ancona con trasparenza e linea continua
        ax.fill(
            ancona_port_polygon_x,
            ancona_port_polygon_y,
            color='black',
            alpha=0.3,  # 70% trasparente (1 - 0.3 = 70%)
            label='Porto di Ancona'
        )
        # Aggiungi il contorno del poligono con una linea continua
        ax.plot(
            ancona_port_polygon_x,
            ancona_port_polygon_y,
            color='black',
            linestyle='-',
            linewidth=1
        )
        # Aggiungi marker per l'antenna AIS
        ax.scatter(ais_antenna_x, ais_antenna_y, color='blue', label='Antenna AIS', s=100, marker='^')

        # Aggiungi la legenda
        ax.legend(loc='upper right')

        # Titolo e salvataggio della figura
        ax.set_title(f'Heatmap of Vessel Positions near Ancona (Date: {date})')
        plt.colorbar(scatter, ax=ax, label='Density')
        plt.tight_layout()
        plt.savefig(os.path.join(year_dir, f'heatmap_ancona_{date}.png'))
        plt.close()


if __name__ == '__main__':
    mp.freeze_support()  # Necessario per Windows
    mp.set_start_method('spawn')  # Compatibilità con Windows

    results_dir = create_unique_directory()

    dataset_folder = r'dataset/AIS_Dataset_csv'  # Assicurati che il percorso sia corretto
    csv_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]
    csv_files = csv_files[:1]

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

    # Verifica se le colonne 'Distance' e 'Bearing' esistono, altrimenti le calcola
    if 'Distance' not in data.columns or 'Bearing' not in data.columns:
        print("Calculating Distance and Bearing...")
        # Supponendo che i dati siano ordinati per nave e per tempo
        data.sort_values(by=['MMSI', 'datetime'], inplace=True)
        data['prev_Latitude'] = data.groupby('MMSI')['Latitude'].shift()
        data['prev_Longitude'] = data.groupby('MMSI')['Longitude'].shift()

        # Calcolo della distanza e dell'angolo di rotta
        def haversine(lon1, lat1, lon2, lat2):
            R = 6371000  # Raggio della Terra in metri
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)

            a = np.sin(delta_phi / 2) ** 2 + \
                np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            meters = R * c
            return meters

        data['Distance'] = haversine(
            data['prev_Longitude'], data['prev_Latitude'],
            data['Longitude'], data['Latitude']
        )

        data['Bearing'] = np.degrees(np.arctan2(
            np.sin(np.radians(data['Longitude'] - data['prev_Longitude'])) * np.cos(np.radians(data['Latitude'])),
            np.cos(np.radians(data['prev_Latitude'])) * np.sin(np.radians(data['Latitude'])) -
            np.sin(np.radians(data['prev_Latitude'])) * np.cos(np.radians(data['Latitude'])) *
            np.cos(np.radians(data['Longitude'] - data['prev_Longitude']))
        ))

        # Pulisce valori infiniti o NaN
        data['Distance'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['Bearing'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(subset=['Distance', 'Bearing'], inplace=True)

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
