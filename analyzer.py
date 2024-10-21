import pandas as pd
import os
from tqdm import tqdm


def analyze_ais_dataset(dataset_path):
    # Inizializza i conteggi e le strutture dati
    unique_mmsi = set()
    mmsi_repeated_count = 0
    total_distance = 0.0
    type_counts = set()

    # Ottieni tutti i file CSV nella cartella specificata
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith('.csv')]

    # Crea una barra di avanzamento con tqdm
    with tqdm(total=len(csv_files), desc="Processing CSV files") as pbar:
        for file in csv_files:
            file_path = os.path.join(dataset_path, file)
            try:
                # Leggi i dati da ogni file CSV
                df = pd.read_csv(file_path)

                # Aggiorna i conteggi unici di MMSI
                unique_mmsi.update(df['MMSI'].unique())

                # Trova MMSI che si ripetono consecutivamente per almeno 10 volte in un giorno
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                df['Date'] = df['Timestamp'].dt.date  # Estrai la data
                df['MMSI_Consecutive'] = (df['MMSI'] != df['MMSI'].shift()).cumsum()  # Gruppo per cambiamento di MMSI

                repeated_mmsi = df.groupby(['Date', 'MMSI_Consecutive']).size()
                mmsi_repeated_count += repeated_mmsi[repeated_mmsi >= 10].count()

                # Somma le distanze
                total_distance += df['Distance'].sum()

                # Aggiorna i type unici
                type_counts.update(df['Type'].dropna().unique())

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

            # Aggiorna la barra di avanzamento
            pbar.update(1)

    # Risultati finali
    print(f"Total unique MMSI: {len(unique_mmsi)}")
    print(f"MMSI repeated consecutively for at least 10 times in a day: {mmsi_repeated_count}")
    print(f"Total average Distance: {total_distance / len(csv_files) if len(csv_files) > 0 else 0}")
    print(f"Total unique Types: {len(type_counts)}")


# Percorso della cartella del dataset
dataset_path = "dataset/AIS_Dataset_csv"
analyze_ais_dataset(dataset_path)
