"""
Script per identificare i vascelli che hanno tracciati validi con piu o meno di 10 chiamate in tutto il dataset
@autore: Andrea Fiorani
"""
import os
import shutil
import pandas as pd
from tqdm import tqdm

source_folder = "dataset/AIS_Tracks_csv"
under10_folder = "dataset/AIS_Tracks_under10_csv"
over10_folder = "dataset/AIS_Tracks_over10_csv"


def main():
    # Creazione delle directory di destinazione se non esistono
    os.makedirs(under10_folder, exist_ok=True)
    os.makedirs(over10_folder, exist_ok=True)

    # Ottieni una lista di tutti i file nel percorso sorgente
    files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    # Processo con barra di avanzamento
    for file in tqdm(files, desc="Processamento dei file", unit="file"):
        file_path = os.path.join(source_folder, file)
        df = pd.read_csv(file_path)
        num_rows = len(df)

        # Copia in base al numero di righe
        if num_rows <= 10:
            dest_path = os.path.join(under10_folder, file)
        else:
            dest_path = os.path.join(over10_folder, file)

        shutil.copy(file_path, dest_path)

    tqdm.write("Suddivisione completata!")


if __name__ == "__main__":
    main()
