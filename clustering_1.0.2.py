import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Usa backend non interattivo per prevenire errori legati a Tcl/Tk

# Importazioni aggiuntive per il clustering e la scalatura
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    Birch,
    DBSCAN,
    MeanShift,
    AgglomerativeClustering,
    SpectralClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Importazione per conversione coordinate geografiche (opzionale)
import pyproj
from sklearn.neighbors import NearestNeighbors

# Import delle metriche di valutazione per il clustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def create_unique_directory(
        base_path="results",
        prefix=f"clustering_{os.path.splitext(os.path.basename(sys.argv[0]))[0].split('_')[-1]}"
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

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcola la distanza in chilometri tra due punti sulla Terra specificati in gradi."""
    R = 6371  # Raggio della Terra in chilometri
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

if __name__ == '__main__':
    # Creazione di una cartella per salvare i risultati
    output_dir = create_unique_directory()

    dataset_folder = "dataset/AIS_Dataset_csv_FocusArea"
    csv_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]

    #csv_files = csv_files[:10]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the folder {dataset_folder}. Please check the path.")

    # Definisci le caratteristiche
    features = ['Latitude', 'Longitude', 'Distance', 'Bearing']

    # Inizializza una lista per memorizzare i dati
    data_list = []

    # Leggi i file CSV e raccogli i dati
    print("Caricamento dei dati...")
    for file in tqdm(csv_files, desc="Caricamento dati"):
        for chunk in pd.read_csv(file, usecols=features, chunksize=100000):
            data_list.append(chunk)

    # Concatena tutti i dati in un unico DataFrame
    df = pd.concat(data_list, ignore_index=True)
    del data_list  # Libera memoria

    # Campionamento dei dati per rendere il calcolo gestibile (opzionale)
    sample_size = 100000  # Modifica questo valore in base alle tue risorse
    df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Preprocessing
    print("Preprocessing dei dati...")
    # Rimuovi eventuali valori NaN o duplicati
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Converti le coordinate geografiche in un sistema di coordinate proiettato (opzionale)
    # Utilizziamo UTM Zone 33N come esempio
    proj = pyproj.Proj(proj='utm', zone=33, ellps='WGS84', preserve_units=False)
    df['Easting'], df['Northing'] = proj(df['Longitude'].values, df['Latitude'].values)

    # Seleziona le caratteristiche per il clustering
    clustering_features = ['Easting', 'Northing', 'Distance', 'Bearing']

    # Scalatura delle caratteristiche
    scaler = StandardScaler()
    X = scaler.fit_transform(df[clustering_features])

    # Salvataggio dello scaler per futuri utilizzi
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvato in {scaler_path}")

    # Riduzione della dimensionalità per visualizzazione e analisi
    print("Calcolo della PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Salvataggio delle componenti principali
    pca_components = pd.DataFrame(X_pca, columns=['Component 1', 'Component 2'])
    pca_components.to_csv(os.path.join(output_dir, "pca_components.csv"), index=False)
    print(f"Componenti principali salvate in {os.path.join(output_dir, 'pca_components.csv')}")

    # Salvataggio dei carichi delle componenti
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_df = pd.DataFrame(loadings, index=clustering_features, columns=['Component 1', 'Component 2'])
    loading_df.to_csv(os.path.join(output_dir, "pca_loadings.csv"))
    print(f"Carichi delle componenti salvati in {os.path.join(output_dir, 'pca_loadings.csv')}")

    # Definizione degli algoritmi di clustering
    clustering_algorithms = {
        'MeanShift': MeanShift(),
    }

    # Esecuzione del clustering e salvataggio dei risultati
    for name, algorithm in clustering_algorithms.items():
        print(f"\nEsecuzione del clustering: {name}")

        # Creazione di una cartella specifica per l'algoritmo
        clustering_output_dir = os.path.join(output_dir, name)
        if not os.path.exists(clustering_output_dir):
            os.makedirs(clustering_output_dir)

        # Per alcuni algoritmi lenti, riduci la dimensione del dataset se necessario
        if name in ['MeanShift', 'SpectralClustering', 'AgglomerativeClustering']:
            sample_size_algo = 5000  # Puoi modificare questo valore
            sample_indices = np.random.choice(len(X), size=sample_size_algo, replace=False)
            X_algo = X[sample_indices]
            X_pca_algo = X_pca[sample_indices]
            df_algo = df.iloc[sample_indices].copy()
        else:
            X_algo = X
            X_pca_algo = X_pca
            df_algo = df.copy()

        # Addestramento del modello di clustering
        if name in ['GaussianMixture']:
            algorithm.fit(X_algo)
            labels = algorithm.predict(X_algo)
        else:
            algorithm.fit(X_algo)
            if hasattr(algorithm, 'labels_'):
                labels = algorithm.labels_
            else:
                labels = algorithm.predict(X_algo)

        # Aggiunta delle etichette al DataFrame
        df_algo['Cluster'] = labels

        # Salvataggio dei risultati
        clustered_data_path = os.path.join(clustering_output_dir, f"{name}_clustered_data.csv")
        df_algo.to_csv(clustered_data_path, index=False)
        print(f"Dati clusterizzati salvati in {clustered_data_path}")

        # Calcolo delle metriche di valutazione (solo se ci sono almeno 2 cluster distinti)
        unique_labels = np.unique(labels)
        n_labels = len(unique_labels)

        if n_labels > 1 and n_labels < len(X_algo):
            silhouette = silhouette_score(X_algo, labels)
            calinski = calinski_harabasz_score(X_algo, labels)
            davies = davies_bouldin_score(X_algo, labels)
        else:
            # Non è possibile calcolare le metriche se c'è solo 1 cluster
            silhouette = None
            calinski = None
            davies = None

        # Salvataggio delle metriche in un CSV
        metrics_dict = {
            'Silhouette Score': [silhouette],
            'Calinski-Harabasz Index': [calinski],
            'Davies-Bouldin Index': [davies]
        }
        metrics_df = pd.DataFrame(metrics_dict)
        metrics_path = os.path.join(clustering_output_dir, f"{name}_metrics.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metriche di clustering salvate in {metrics_path}")

        # Visualizzazione dei cluster (PCA)
        plt.figure(figsize=(10, 8))
        colors = plt.get_cmap('tab20', lut=n_labels)

        for label in unique_labels:
            class_member_mask = (labels == label)
            xy = X_pca_algo[class_member_mask]
            if label == -1:
                # Rumore in DBSCAN
                plt.scatter(xy[:, 0], xy[:, 1], s=10, c='k', marker='x', label='Noise')
            else:
                plt.scatter(xy[:, 0], xy[:, 1], s=10, color=colors(label), label=f'Cluster {label}')

        plt.title(f"Clustering dei dati - {name}")
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.legend()
        plt.savefig(os.path.join(clustering_output_dir, f"{name}_clusters.png"))
        plt.close()
        print(f"Visualizzazione dei cluster salvata come {os.path.join(clustering_output_dir, f'{name}_clusters.png')}")

        # Salvataggio del modello di clustering (se possibile)
        model_path = os.path.join(clustering_output_dir, f"{name}_model.pkl")
        try:
            joblib.dump(algorithm, model_path)
            print(f"Modello di clustering salvato in {model_path}")
        except Exception as e:
            print(f"Impossibile salvare il modello per {name}: {e}")

        # Analisi dei cluster
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        cluster_counts_path = os.path.join(clustering_output_dir, f"{name}_cluster_counts.csv")
        cluster_counts.to_csv(cluster_counts_path)
        print(f"Conteggio dei cluster salvato in {cluster_counts_path}")

    print("Clustering completato.")
