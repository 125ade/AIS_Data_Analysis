"""
Script per la classificazione delle tipologie delle navi identificate
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
matplotlib.use('Agg')


def create_unique_directory(
        base_path="results",
        prefix=f"classification_{os.path.splitext(os.path.basename(sys.argv[0]))[0].split('_')[-1]}"
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


if __name__ == '__main__':
    output_dir = create_unique_directory()

    dataset_folder = "dataset/AIS_Dataset_csv_FocusArea"
    csv_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in the folder {dataset_folder}. Please check the path.")

    # Definisci le caratteristiche e la variabile target
    features = ['Latitude', 'Longitude', 'Distance', 'Bearing']
    target = 'Type'

    # Ottieni le classi presenti nel dataset
    print("Calcolo delle classi presenti nel dataset...")
    class_counts = {}
    total_rows = 0
    # Legge i file CSV e calcola la distribuzione delle classi
    for file in tqdm(csv_files, desc="Calcolo delle classi"):
        for chunk in pd.read_csv(file, usecols=[target], chunksize=100000):
            total_rows += len(chunk)
            chunk_counts = chunk[target].value_counts().to_dict()
            for cls, count in chunk_counts.items():
                class_counts[cls] = class_counts.get(cls, 0) + count

    # Calcola il numero di classi e il numero di campioni per classe
    classes_to_remove = [11, 19]
    classes = [cls for cls in class_counts.keys() if cls not in classes_to_remove]
    num_classes = len(classes)
    samples_per_class = 30000 // num_classes

    print(f"Numero di classi: {num_classes}")
    print(f"Campioni per classe: {samples_per_class}")

    # Inizializza un dizionario per tenere traccia dei campioni raccolti per classe
    collected_samples = {cls: 0 for cls in classes}

    # Inizializza una lista per memorizzare i campioni raccolti
    collected_data = []

    # Leggi i file CSV e raccogli i campioni necessari per ciascuna classe
    print("Raccolta dei campioni per creare il dataset bilanciato...")
    for file in tqdm(csv_files, desc="Raccolta campioni"):
        for chunk in pd.read_csv(file, usecols=features + [target], chunksize=100000):
            # Filtra solo le classi che necessitano ancora di campioni
            needed_classes = [cls for cls, count in collected_samples.items() if count < samples_per_class]
            chunk = chunk[chunk[target].isin(needed_classes)]

            # Per ogni classe, raccogli i campioni necessari
            for cls in needed_classes:
                cls_data = chunk[chunk[target] == cls]
                needed = samples_per_class - collected_samples[cls]
                if needed > 0 and not cls_data.empty:
                    samples_to_collect = min(len(cls_data), needed)
                    sampled_data = cls_data.sample(n=samples_to_collect, random_state=42)
                    collected_data.append(sampled_data)
                    collected_samples[cls] += samples_to_collect

            # Verifica se abbiamo raccolto tutti i campioni necessari
            if all(count >= samples_per_class for count in collected_samples.values()):
                break  # Esci dal ciclo dei chunk

        # Verifica se abbiamo raccolto tutti i campioni necessari
        if all(count >= samples_per_class for count in collected_samples.values()):
            break  # Esci dal ciclo dei file CSV

    # Concatena tutti i campioni raccolti in un unico DataFrame
    df = pd.concat(collected_data, ignore_index=True)
    del collected_data  # Libera memoria

    # Shuffle del dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Seleziona le caratteristiche e la variabile target
    X = df[features]
    y = df[target]

    print("Creazione dei set di training e test...")
    # Dividi il dataset in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Distribuzione delle classi nel set di training:")
    print(y_train.value_counts())
    print("\nDistribuzione delle classi nel set di test:")
    print(y_test.value_counts())

    # Calcolo dei pesi delle classi
    classes_array = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced', classes=classes_array, y=y_train
    )
    class_weights_dict = dict(zip(classes_array, class_weights))

    # Scalatura delle caratteristiche
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Salvataggio dello scaler per futuri utilizzi
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler salvato in {scaler_path}")

    # Definizione dei classificatori
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42, class_weight=class_weights_dict),
        'DecisionTree': DecisionTreeClassifier(random_state=42, class_weight=class_weights_dict),
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }

    # Addestramento e valutazione dei modelli
    for name, model in classifiers.items():
        print(f"\nAddestramento del modello: {name}")

        # Creazione di una cartella specifica per il modello
        model_output_dir = os.path.join(output_dir, name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        # Verifica se il modello richiede dati scalati
        if name in ['LogisticRegression']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test

        # Addestramento del modello
        model.fit(X_train_model, y_train)

        # Predizioni
        y_pred = model.predict(X_test_model)

        # Calcolo delle metriche
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # Salvataggio delle metriche
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(os.path.join(model_output_dir, f"{name}_classification_report.csv"), index=True)
        print(f"Classification Report salvato in {os.path.join(model_output_dir, f'{name}_classification_report.csv')}")

        # Salvataggio della Confusion Matrix come immagine
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes_array, yticklabels=classes_array
        )
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(model_output_dir, f"{name}_confusion_matrix.png"))
        plt.close()
        print(f"Confusion Matrix salvata come {os.path.join(model_output_dir, f'{name}_confusion_matrix.png')}")

        # Salvataggio del modello addestrato
        model_path = os.path.join(model_output_dir, f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Modello salvato in {model_path}")

        # Salvataggio di altre informazioni
        with open(os.path.join(model_output_dir, f"{name}_metrics.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Confusion Matrix:\n{conf_matrix}\n")
            f.write("Classification Report:\n")
            f.write(report_df.to_string())
        print(f"Metriche salvate in {os.path.join(model_output_dir, f'{name}_metrics.txt')}")
