"""
Script che crea file separati per ogni MMSI nel dataset di partenza
"""
import pandas as pd
import os
from tqdm import tqdm
import psutil
import argparse


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
        tqdm.write("Nessun dato AIS valido trovato.")
        return None

    # Creare un dizionario per contenere le serie temporali per ogni vascello
    vessel_tracks = {}

    # Scorrere ogni MMSI unico nel dataset
    for mmsi in tqdm(df['MMSI'].unique(), desc="Estrazione tracciati per MMSI", unit=" MMSI"):
        vessel_data = df[df['MMSI'] == mmsi]

        # Verificare che esistano più di un punto di rilevamento per creare una serie temporale
        if len(vessel_data) > 1:
            vessel_tracks[mmsi] = vessel_data[['id', 'timestamp', 'Latitude', 'Longitude', 'Distance', 'Bearing', 'Type']]

    if not vessel_tracks:
        tqdm.write("Non sono stati trovati tracciati validi per i vascelli.")
    else:
        tqdm.write(f"Tracciati trovati per {len(vessel_tracks)} vascelli.")

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
        tqdm.write(f"Cartella creata: {directory}")


# Funzione principale
def main():
    parser = argparse.ArgumentParser(description="Processa dati AIS e salva tracciati per vascelli.")
    parser.add_argument('--year', type=str, required=True, help='Anno da selezionare (es. 2020)')
    args = parser.parse_args()
    selected_year = args.year

    # Ottieni una lista di tutti i file nella directory specificata che contengono l'anno selezionato
    csv_directory = 'dataset/AIS_Dataset_csv'
    csv_files = [f for f in os.listdir(csv_directory) if selected_year in f]

    # Verifica se ci sono file CSV trovati
    if not csv_files:
        tqdm.write(f"Nessun file CSV trovato nella directory '{csv_directory}' per l'anno {selected_year}.")
        exit()

    tqdm.write(f"Anno selezionato: {selected_year}")

    # Prepara i percorsi completi dei file
    csv_files_full_path = [os.path.join(csv_directory, f) for f in csv_files]

    # Mostra barra di avanzamento per il caricamento dei dati
    tqdm.write(f"\nCaricamento dei file CSV per l'anno {selected_year}...")
    df = load_csv_files(csv_files_full_path)

    # Mostra l'uso della memoria RAM dopo il caricamento dei file
    ram_usage = memory_usage()
    tqdm.write(f"\nUso della memoria RAM dopo il caricamento dei file: {ram_usage:.2f} MB")

    # Crea la cartella per salvare i tracciati, se non esiste
    tracks_directory = 'dataset/AIS_Tracks_csv'
    create_directory_if_not_exists(tracks_directory)

    # Estrapola le time series per ogni vascello
    tqdm.write("\nEstrazione delle time series per ciascun vascello...")
    vessel_tracks = extract_vessel_tracks(df)

    # Verifica e salva i tracciati trovati
    if vessel_tracks:
        tqdm.write("\nSalvataggio tracciati per MMSI in corso...")
        saved_count = 0
        skipped_count = 0
        for mmsi, track in tqdm(vessel_tracks.items(), desc="Salvataggio tracciati", unit=" tracciati"):
            file_path = os.path.join(tracks_directory, f'vessel_track_{selected_year}_{mmsi}.csv')  # Includi l'anno nel nome del file
            if os.path.exists(file_path):
                skipped_count += 1
                continue
            try:
                track.to_csv(file_path, index=False)
                saved_count += 1
            except Exception as e:
                tqdm.write(f"Errore nel salvare il file {file_path}: {e}")

        tqdm.write(f"\nSalvataggio completato. Tracciati salvati: {saved_count}, Tracciati saltati: {skipped_count}.")
    else:
        tqdm.write("\nNessun tracciato da salvare.")


if __name__ == "__main__":
    main()
