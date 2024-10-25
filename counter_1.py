"""
Script for counting various metrics on the dataset to verify the creation of valid datasets.
@author: Andrea Fiorani
"""
import os
from tqdm import tqdm

# Definisci i percorsi e gli anni da cercare
paths = ["dataset/AIS_Dataset", "dataset/AIS_Dataset_csv", "dataset/AIS_Tracks_csv", "dataset/AIS_Tracks_over10_csv", "dataset/AIS_Tracks_under10_csv"]
years = ["2020", "2021", "2022", "2023"]


# Funzione per contare i file contenenti un anno nel nome in un percorso
def count_files_by_year_in_path(path, years):
    year_counts = {year: 0 for year in years}
    file_names = []
    if os.path.exists(path) and os.path.isdir(path):
        files = [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]

        # Conteggio e visualizzazione barra di avanzamento
        for file in tqdm(files, desc=f"Contando file in {path}", unit="file"):
            for year in years:
                if year in file:
                    year_counts[year] += 1
            file_names.append(file)

        # Mostra i nomi dei file se ce ne sono meno di 10
        if len(files) < 10:
            print(f"Nomi dei file in '{path}': {files}")
    else:
        print(f"Percorso non trovato o non è una directory: {path}")

    return year_counts, len(files)


# Conteggio file per ogni percorso
path_year_counts = {}
path_file_counts = {}
for path in paths:
    counts, total_files = count_files_by_year_in_path(path, years)
    path_year_counts[path] = counts
    path_file_counts[path] = total_files

# Stampa il risultato
for path, counts in path_year_counts.items():
    print(f"\nConteggio file in '{path}':")
    for year, count in counts.items():
        print(f"  File con '{year}' nel nome: {count}")
    print(f"  Totale file: {path_file_counts[path]}")

