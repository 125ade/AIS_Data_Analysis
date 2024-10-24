import pandas as pd
import os
from tqdm import tqdm
import psutil  # Libreria per monitorare l'uso della memoria RAM


# Funzione per caricare e unire i dati da più file CSV
def load_csv_files(file_paths):
    dataframes = []
    for file_path in tqdm(file_paths, desc="Caricamento CSV", unit=" file"):  # Barra di avanzamento per i file
        df = pd.read_csv(file_path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


# Funzione per estrapolare le time series per ciascun vascello (MMSI)
def extract_vessel_tracks(df):
    # Verifica che il dataframe contenga dati validi
    if df.empty or 'MMSI' not in df.columns:
        print("Nessun dato AIS valido trovato.")
        return None

    # Creare un dizionario per contenere le serie temporali per ogni vascello
    vessel_tracks = {}

    # Scorrere ogni MMSI unico nel dataset
    for mmsi in tqdm(df['MMSI'].unique(), desc="Estrazione tracciati per MMSI", unit=" MMSI"):
        vessel_data = df[df['MMSI'] == mmsi]

        # Verificare che esistano più di un punto di rilevamento per creare una serie temporale
        if len(vessel_data) > 1:
            # Includiamo anche la colonna 'id' nel tracciato del vascello
            vessel_tracks[mmsi] = vessel_data[['id', 'timestamp', 'Latitude', 'Longitude', 'Distance', 'Bearing', 'Type']]

    if not vessel_tracks:
        print("Non sono stati trovati tracciati validi per i vascelli.")
    else:
        print(f"Tracciati trovati per {len(vessel_tracks)} vascelli.")

    return vessel_tracks


# Funzione per monitorare l'uso della memoria
def memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / (1024 ** 2)  # Convertire in MB
    return mem_mb


# Funzione per creare una cartella se non esiste
def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Cartella creata: {directory}")


# Lista dei primi 5 file CSV
# csv_files = [f'dataset/AIS_Dataset_csv/ais_data_2020_p_{i}.csv' for i in range(1, 108)]
# Ottieni una lista di tutti i file nella directory specificata
csv_files = [f for f in os.listdir('dataset/AIS_Dataset_csv') if '2020' in f]

# Prepara i percorsi completi dei file
csv_files_full_path = [os.path.join('dataset/AIS_Dataset_csv', f) for f in csv_files]

X: int = 10

# Mostra barra di avanzamento per il caricamento dei dati
print(f"\nCaricamento dei {f'primi {X} ' if X == 10 else ''}file CSV...")
df = load_csv_files(csv_files_full_path[:X])

# Mostra l'uso della memoria RAM dopo il caricamento dei file
ram_usage = memory_usage()
print(f"\nUso della memoria RAM dopo il caricamento dei file: {ram_usage:.2f} MB")

# Crea la cartella per salvare i tracciati, se non esiste
tracks_directory = 'dataset/AIS_Tracks_csv'
create_directory_if_not_exists(tracks_directory)

# Estrapola le time series per ogni vascello
print("\nEstrazione delle time series per ciascun vascello...")
vessel_tracks = extract_vessel_tracks(df)

# Verifica e salva i tracciati trovati
if vessel_tracks:
    print("\nSalvataggio tracciati per MMSI in corso...")
    for mmsi, track in tqdm(vessel_tracks.items(), desc="Salvataggio tracciati", unit=" tracciati"):
        file_path = os.path.join(tracks_directory, f'vessel_track_{mmsi}.csv')
        track.to_csv(file_path, index=False)
