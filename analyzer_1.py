"""
Script to analyze AIS tracks and generate charts
- distribution of the number of detections,
- days of the week,
- hours of the day,
- boat type,
- calculate the variance of the time delta between detections.
@author: Andrea Fiorani
"""
import pandas as pd
import os
import sys
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

matplotlib.use('Agg')


# Funzione per creare una cartella unica "analysis_n" all'interno di "results"
def create_unique_directory(base_path="results",
                            prefix=f"analysis_{os.path.splitext(os.path.basename(sys.argv[0]))[0].split('_')[-1]}"):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Trova il numero univoco per la cartella
    n = 1
    while os.path.exists(os.path.join(base_path, f"{prefix}_{n}")):
        n += 1

    unique_directory = os.path.join(base_path, f"{prefix}_{n}")
    os.makedirs(unique_directory)
    return unique_directory


# Funzione per calcolare la varianza del delta t tra le rilevazioni e salvare i grafici
def plot_delta_t_variance(tracks_directory, save_path):
    track_files = [f for f in os.listdir(tracks_directory) if f.endswith('.csv')]
    delta_t_variances = []  # Lista per memorizzare le varianze del delta t per ciascun file
    mean_delta_t = []  # Lista per memorizzare il delta t medio per ciascun file

    for track_file in tqdm(track_files, desc="Calcolo varianza delta t per ogni file", unit=" file"):
        file_path = os.path.join(tracks_directory, track_file)

        # Carica il tracciato
        track_df = pd.read_csv(file_path)

        # Controlla se ci sono abbastanza dati per calcolare la varianza
        if len(track_df) < 2:
            continue  # Salta il file se ha meno di 2 rilevazioni

        # Calcola le differenze temporali
        track_df['timestamp'] = pd.to_datetime(track_df['timestamp'], unit='s')
        track_df = track_df.sort_values(by='timestamp')
        time_diffs = track_df['timestamp'].diff().dropna().dt.total_seconds().abs()

        # Calcola la varianza e il delta t medio
        variance = time_diffs.var()
        mean_dt = time_diffs.mean()

        if pd.notna(variance) and mean_dt > 0:
            delta_t_variances.append(variance)
            mean_delta_t.append(mean_dt)

    # Crea un dataframe per analizzare i dati
    variance_df = pd.DataFrame({
        'Mean Delta T (s)': mean_delta_t,
        'Variance Delta T (s^2)': delta_t_variances
    })

    # Lista dei fattori di conversione e delle unità con tutte le soglie richieste
    time_units = [
        ('seconds', 1,
         [300, 600, 1200, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 604800, 2592000],
         ['5 min', '10 min', '20 min', '30 min', '1 hour', '2 hours', '3 hours', '6 hours', '12 hours', '1 day', '2 days', '1 week', '1 month'],
         ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'grey', 'black', 'pink', 'lightblue', 'teal']),
        ('minutes', 60,
         [5, 10, 20, 30, 60, 120, 180, 360, 720, 1440, 2880, 10080, 43200],
         ['5 min', '10 min', '20 min', '30 min', '1 hour', '2 hours', '3 hours', '6 hours', '12 hours', '1 day', '2 days', '1 week', '1 month'],
         ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'grey', 'black', 'pink', 'lightblue', 'teal']),
        ('hours', 3600,
         [1, 2, 3, 6, 12, 24, 48, 168, 720],
         ['1 hour', '2 hours', '3 hours', '6 hours', '12 hours', '1 day', '2 days', '1 week', '1 month'],
         ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'grey'])
    ]

    # Soglie da evidenziare in secondi, minuti e ore
    for unit_name, unit_factor, thresholds_unit, threshold_labels, colors in time_units:
        # Conversione dei dati nell'unità di tempo desiderata
        variance_df_unit = variance_df.copy()
        variance_df_unit['Mean Delta T'] = variance_df_unit['Mean Delta T (s)'] / unit_factor

        # Grafico della varianza del delta t con scala logaritmica sull'asse x
        plt.figure(figsize=(10, 6))
        plt.scatter(
            variance_df_unit['Mean Delta T'],
            variance_df_unit['Variance Delta T (s^2)'],
            alpha=0.6,
            edgecolors='w',
            s=50
        )
        plt.xscale('log')  # Imposta la scala logaritmica sull'asse x
        plt.xlabel(f"Average Delta T between measurements ({unit_name}) [Log Scale]")
        plt.ylabel("Variance of Delta T (s^2)")
        plt.title(f"Variance of Delta T vs Average Delta T between measurements ({unit_name})")
        plt.grid(True, which="both", ls="--", linewidth=0.5)

        # Inizializza una lista per gli elementi della legenda
        legend_elements = []

        # Evidenzia soglie opportune e raccogli i conteggi per la legenda
        for threshold, label, color in zip(thresholds_unit, threshold_labels, colors):
            plt.axvline(x=threshold, color=color, linestyle='--', linewidth=1)

            # Calcola i conteggi
            count_less = (variance_df_unit['Mean Delta T'] < threshold).sum()
            count_equal = (variance_df_unit['Mean Delta T'] == threshold).sum()
            count_greater = (variance_df_unit['Mean Delta T'] > threshold).sum()

            # Crea un'etichetta con i conteggi
            label_with_counts = f"{label}: <{count_less}, ={count_equal}, >{count_greater}"

            # Crea un handle per la legenda
            handle = Line2D([0], [0], color=color, linestyle='--', linewidth=1)
            legend_elements.append((handle, label_with_counts))

            # Aggiungi il testo al grafico
            plt.text(threshold, plt.ylim()[1] * 0.9, label, color=color, rotation=90, va='top', ha='right')

        # Crea la legenda personalizzata
        handles, labels = zip(*legend_elements)
        plt.legend(handles, labels, title="Thresholds", bbox_to_anchor=(1.05, 1), loc='upper left')

        # Salva il grafico
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"delta_t_variance_{unit_name}.png"))
        plt.close()


# Funzione per generare i grafici di distribuzione per numero di rilevazioni, giorni della settimana, ore del giorno, tipologia di barca e salvarli
def analyze_vessel_tracks(tracks_directory, save_path):
    track_files = [f for f in os.listdir(tracks_directory) if f.endswith('.csv')]

    rows_per_file = {}
    types_per_file = {}
    weekdays = []
    hours = []
    week_info = []  # Lista per memorizzare settimana, mese, anno, giorno della settimana

    for track_file in tqdm(track_files, desc="Analisi file per grafici di distribuzione", unit=" file"):
        mmsi = track_file.split('_')[-1].split('.')[0]
        file_path = os.path.join(tracks_directory, track_file)

        # Carica il tracciato
        track_df = pd.read_csv(file_path)

        # Numero di righe (rilevazioni) per file
        rows_per_file[mmsi] = len(track_df)

        # Estrazione della tipologia di barca (presumendo un unico tipo per file)
        if 'Type' in track_df.columns:
            boat_type = track_df['Type'].mode()[0] if not track_df['Type'].mode().empty else 'Unknown'
        else:
            boat_type = 'Unknown'
        types_per_file[mmsi] = boat_type

        # Conversione del timestamp in datetime e estrazione di giorni e ore
        track_df['timestamp'] = pd.to_datetime(track_df['timestamp'], unit='s')
        weekdays.extend(track_df['timestamp'].dt.day_name())
        hours.extend(track_df['timestamp'].dt.hour)

        # Estrazione settimana, mese, anno per ogni rilevazione
        track_df = track_df.sort_values(by='timestamp')
        isocalendar = track_df['timestamp'].dt.isocalendar()
        track_df['week'] = isocalendar.week
        track_df['year'] = isocalendar.year
        track_df['month'] = track_df['timestamp'].dt.month

        for _, row in track_df.iterrows():
            week_info.append({
                'week': row['week'],
                'year': row['year'],
                'month': row['month'],
                'weekday': row['timestamp'].day_name()
            })

    # Crea un dataframe per la distribuzione della tipologia di barca
    distribution_df = pd.DataFrame({
        'MMSI': list(rows_per_file.keys()),
        'Boat Type': list(types_per_file.values()),
        'Number of Detections': list(rows_per_file.values())
    })

    # Crea un dataframe per le informazioni settimanali
    week_df = pd.DataFrame(week_info)

    # Grafico della distribuzione del numero di rilevazioni per file (MMSI) con scala logaritmica sull'asse y
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(rows_per_file)), rows_per_file.values(), color='skyblue')
    plt.xlabel("File (MMSI)")
    plt.yscale("log")  # Imposta la scala logaritmica sull'asse y
    plt.ylabel("Number of measurements")
    plt.title("Distribution of the number of measurements for each MMSI")

    # Aggiunge righe orizzontali a y=5, y=10, y=20, y=30, y=100 con i numeri affiancati e conta i file sopra ogni soglia
    thresholds = [5, 10, 20, 30, 100]
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    counts_above_thresholds = [sum([v > t for v in rows_per_file.values()]) for t in thresholds]

    for i, threshold in enumerate(thresholds):
        plt.axhline(y=threshold, color=colors[i], linestyle='--', linewidth=0.7)
        plt.text(x=0.02 * len(rows_per_file), y=threshold, s=f"{threshold}", color=colors[i], va='center', ha='left')
        plt.text(x=0.98 * len(rows_per_file), y=threshold, s=f"{counts_above_thresholds[i]} file > {threshold}",
                 color=colors[i], va='center', ha='right')

    # Rimuove i nomi dei file dall'asse x
    plt.xticks(ticks=[])

    # Salva il grafico
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "distribution_rows_per_file.png"))
    plt.close()

    # Grafico della distribuzione dei giorni della settimana
    plt.figure(figsize=(10, 6))
    pd.Series(weekdays).value_counts().sort_index().plot(kind='bar', color='salmon')
    plt.xlabel("Day of the week")
    plt.ylabel("Number of measurements")
    plt.title("Distribution of measurements by day of the week")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "distribution_weekdays.png"))
    plt.close()

    # Grafico della distribuzione oraria delle rilevazioni
    plt.figure(figsize=(10, 6))
    pd.Series(hours).value_counts().sort_index().plot(kind='bar', color='lightgreen')
    plt.xlabel("Hour of the day")
    plt.ylabel("Number of measurements")
    plt.title("Hourly distribution of measurements")
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "distribution_hours.png"))
    plt.close()

    # Grafico della distribuzione della tipologia di barca in base al numero di rilevazioni
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        x='Boat Type',
        y='Number of Detections',
        hue='Boat Type',  # Assegna 'Boat Type' al parametro hue
        data=distribution_df,
        palette='Set3',
        dodge=False,  # Impedisce lo spostamento dei boxplot
        showfliers=False  # Nasconde i valori anomali per una visualizzazione più pulita
    )
    plt.xlabel("Type of Boat")
    plt.ylabel("Number of measurements")
    plt.title("Distribution of Boat Types by Number of Measurements")
    plt.yscale("log")  # Imposta la scala logaritmica sull'asse y per una migliore visualizzazione
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(title='Boat Type', bbox_to_anchor=(1.05, 1), loc='upper left')  # Sposta la legenda fuori dal grafico
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "distribution_boat_type.png"))
    plt.close()

    # Grafico della distribuzione dei rilevamenti per settimana
    # Crea una cartella "distribution_week"
    distribution_week_path = os.path.join(save_path, "distribution_week")
    os.makedirs(distribution_week_path, exist_ok=True)

    # Raggruppa per anno e settimana
    grouped_weeks = week_df.groupby(['year', 'week'])

    for (year, week), group in grouped_weeks:
        # Conta le rilevazioni per giorno della settimana in questa settimana
        weekday_counts = group['weekday'].value_counts().reindex(
            ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], fill_value=0)

        # Plot
        plt.figure(figsize=(8, 6))
        weekday_counts.plot(kind='bar', color='skyblue')
        plt.xlabel("Day of the week")
        plt.ylabel("Number of measurements")
        plt.title(f"Distribution of measurements for week {week} of {year}")
        plt.tight_layout()

        # Salva il grafico con nome "week{week}_{month}_{year}.png"
        # Poiché una settimana può includere più mesi, scegli il mese più frequente nella settimana
        common_month = group['month'].mode()[0]
        plt.savefig(os.path.join(distribution_week_path, f"{week}_w_{common_month}_m_{year}_y.png"))
        plt.close()


# Percorso della cartella dei tracciati già estratti
tracks_directory = 'dataset/AIS_Tracks_csv'

# Creazione della cartella di salvataggio unica
results_directory = create_unique_directory()

# Generazione e salvataggio dei grafici
plot_delta_t_variance(tracks_directory, results_directory)
analyze_vessel_tracks(tracks_directory, results_directory)
