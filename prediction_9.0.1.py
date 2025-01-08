import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Usa un backend non interattivo per evitare problemi con Tcl
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from prophet import Prophet

# >>> IMPORT PER I TEST DI STAZIONARIETÀ <<<
from statsmodels.tsa.stattools import adfuller, kpss

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------------------------------------------------------
# FUNZIONI DI UTILITÀ
# -------------------------------------------------------------------

def create_unique_directory(base_path="results", prefix=None):
    """Crea una directory unica per salvare i risultati."""
    if prefix is None:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        prefix = f"evaluation_{script_name.split('_')[-1]}"
    tqdm.write(f"Using prefix: {prefix}")

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    n = 1
    while os.path.exists(os.path.join(base_path, f"{prefix}_{n}")):
        n += 1

    unique_directory = os.path.join(base_path, f"{prefix}_{n}")
    os.makedirs(unique_directory)
    return unique_directory


def evaluate_metrics(y_true, y_pred):
    """
    Calcola MAE, RMSE, MAPE tra y_true e y_pred.
    Se y_true ha valori nulli, la parte corrispondente nel MAPE viene ignorata.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = (y_true != 0)
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_prediction(train_df, test_df, predictions, model_name, results_dir):
    """
    Crea un plot dell'intero dataset:
    - Train in blu
    - Test in verde
    - Predizioni in rosso
    E salva in results_dir/plot_{model_name}.png
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_df['date'], train_df['unique_ships'], color='blue', label='Train')
    plt.plot(test_df['date'], test_df['unique_ships'], color='green', label='Test')
    plt.plot(test_df['date'], predictions, color='red', label='Prediction')
    plt.title(f"Modello: {model_name}")
    plt.xlabel('Date')
    plt.ylabel('Unique Ships')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"plot_{model_name}.png"))
    plt.close()


# >>> FUNZIONE PER VERIFICA DI STAZIONARIETÀ <<<
def test_stationarity(df, col_name, results_dir=None):
    """
    Esegue ADF e KPSS sulla colonna `col_name` del DataFrame `df`.
    Riporta:
      - p-value
      - lags used
      - observations
      - critical values (1%, 5%, 10%)
    e salva i risultati in un file di testo in results_dir.
    """
    data = df[col_name].dropna()  # Rimuove eventuali NaN
    if len(data) < 10:
        tqdm.write(f"Non ci sono abbastanza dati per testare la stazionarietà in {col_name}.")
        return

    # ========== ADF ==========
    adf_result = adfuller(data, autolag='AIC')
    adf_stat, adf_pvalue, adf_usedlag, adf_nobs, adf_crit, adf_icbest = adf_result
    # adf_crit è un dizionario con chiavi '1%', '5%', '10%', ecc.

    # ========== KPSS ==========
    kpss_result = kpss(data, regression='c', nlags='auto')
    kpss_stat, kpss_pvalue, kpss_lags, kpss_crit = kpss_result
    # kpss_crit è un dizionario con chiavi '1%', '5%', '10%', ecc.

    out_text = [
        f"=== Test di stazionarietà su '{col_name}' ===",
        f"N. osservazioni (len(data)): {len(data)}",
        "",
        "------- ADF TEST -------",
        f"ADF Statistic: {adf_stat}",
        f"ADF p-value:  {adf_pvalue}",
        f"ADF lags used: {adf_usedlag}",
        f"ADF observations: {adf_nobs}",
        f"ADF critical values:",
        f"    1% -> {adf_crit.get('1%', 'NA')}",
        f"    5% -> {adf_crit.get('5%', 'NA')}",
        f"    10% -> {adf_crit.get('10%', 'NA')}",
        "",
        "------- KPSS TEST ------",
        f"KPSS Statistic: {kpss_stat}",
        f"KPSS p-value:   {kpss_pvalue}",
        f"KPSS lags used: {kpss_lags}",
        f"KPSS critical values:",
        f"    1% -> {kpss_crit.get('1%', 'NA')}",
        f"    5% -> {kpss_crit.get('5%', 'NA')}",
        f"    10% -> {kpss_crit.get('10%', 'NA')}",
        "----------------------------------------",
        "NOTE:",
        " - ADF: ipotesi nulla = non stazionaria (unit root).",
        "   p-value < 0.05 => rifiuto ip. nulla => serie stazionaria.",
        " - KPSS: ipotesi nulla = stazionarietà.",
        "   p-value > 0.05 => non rifiuto stazionarietà."
    ]
    out_text_str = "\n".join(out_text)

    # Stampa su console
    tqdm.write(out_text_str)

    # Salva su file, se results_dir è specificata
    if results_dir:
        fname = os.path.join(results_dir, f"stationarity_{col_name}.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(out_text_str)


# -------------------------------------------------------------------
# CARICAMENTO E PREPROCESSING
# -------------------------------------------------------------------

def load_and_preprocess_data():
    """
    Carica i CSV in 'dataset/AIS_Dataset_csv_FocusArea', aggrega le navi uniche settimanalmente
    e restituisce un DataFrame con colonne ['date', 'unique_ships'].
    """
    dataset_folder = "dataset/AIS_Dataset_csv_FocusArea"
    csv_files = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith('.csv')]

    all_data = []
    for file in tqdm(csv_files, desc="Caricamento dei file CSV"):
        df = pd.read_csv(file, usecols=['timestamp', 'MMSI'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.to_period('W').dt.start_time
        weekly_unique = df.groupby('date')['MMSI'].nunique().reset_index()
        weekly_unique.rename(columns={'MMSI': 'unique_ships'}, inplace=True)
        all_data.append(weekly_unique)

    full_data = pd.concat(all_data)
    full_data = full_data.groupby('date')['unique_ships'].sum().reset_index()
    full_data = full_data.sort_values('date').reset_index(drop=True)

    return full_data


def split_data(df):
    """
    Suddivide il DataFrame in train/test basato su date:
    - train: date < '2023-01-01'
    - test : date >= '2023-01-01'
    """
    train = df[df['date'] < '2023-01-01'].copy()
    test = df[df['date'] >= '2023-01-01'].copy()
    return train, test


# -------------------------------------------------------------------
# CREAZIONE FEATURE LAG
# -------------------------------------------------------------------

def create_features_lag(df, lag=4):
    df = df.copy()
    for i in range(1, lag + 1):
        df[f'lag_{i}'] = df['unique_ships'].shift(i)
    df.dropna(inplace=True)
    return df


# -------------------------------------------------------------------
# ARIMA, SARIMA, SARIMAX - param fissi
# -------------------------------------------------------------------

def train_arima(train_df, test_df, p, d, q):
    """Allena un ARIMA(p,d,q) direttamente senza grid search."""
    tqdm.write(f"Training ARIMA({p},{d},{q})...")
    try:
        model = ARIMA(train_df['unique_ships'], order=(p, d, q))
        fit = model.fit()
        preds_test = fit.forecast(steps=len(test_df))
        aic = fit.aic
        bic = fit.bic
        return preds_test.values, aic, bic
    except Exception as e:
        raise ValueError(f"ARIMA({p},{d},{q}) Error: {str(e)}")


def train_sarima(train_df, test_df, p, d, q, P, D, Q, s):
    """Allena un SARIMA(p,d,q)(P,D,Q,s) senza grid search."""
    tqdm.write(f"Training SARIMA({p},{d},{q})x({P},{D},{Q},{s})...")
    try:
        model = SARIMAX(
            train_df['unique_ships'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s)
        )
        fit = model.fit(disp=False)
        preds_test = fit.forecast(steps=len(test_df))
        aic = fit.aic
        bic = fit.bic
        return preds_test.values, aic, bic
    except Exception as e:
        raise ValueError(f"SARIMA({p},{d},{q})x({P},{D},{Q},{s}) Error: {str(e)}")


def train_sarimax(train_df, test_df, p, d, q, P, D, Q, s):
    """Allena un SARIMAX(p,d,q)(P,D,Q,s) con param fissi."""
    tqdm.write(f"Training SARIMAX({p},{d},{q})x({P},{D},{Q},{s})...")
    try:
        model = SARIMAX(
            train_df['unique_ships'],
            order=(p, d, q),
            seasonal_order=(P, D, Q, s)
        )
        fit = model.fit(disp=False)
        preds_test = fit.forecast(steps=len(test_df))
        aic = fit.aic
        bic = fit.bic
        return preds_test.values, aic, bic
    except Exception as e:
        raise ValueError(f"SARIMAX({p},{d},{q})x({P},{D},{Q},{s}) Error: {str(e)}")


# -------------------------------------------------------------------
# HOLT-WINTERS (mini tuning su trend/seasonal)
# -------------------------------------------------------------------

def train_holt_winters_tuning(train, test):
    vali_size = int(len(train) * 0.1)
    train_sub = train.iloc[:-vali_size]
    vali_sub = train.iloc[-vali_size:]

    combos = [
        ('add', 'add'),
        ('mul', 'add'),
        ('add', 'mul'),
        ('mul', 'mul'),
    ]
    best_rmse = np.inf
    best_combo = None

    for trend, seasonal in tqdm(combos, desc="Testing Holt-Winters combos"):
        try:
            model = ExponentialSmoothing(
                train_sub['unique_ships'],
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=52
            )
            fit = model.fit()
            preds_vali = fit.forecast(steps=len(vali_sub))
            rmse_vali = np.sqrt(mean_squared_error(vali_sub['unique_ships'], preds_vali))
            if rmse_vali < best_rmse:
                best_rmse = rmse_vali
                best_combo = (trend, seasonal)
        except:
            pass

    if best_combo:
        tqdm.write(f"Holt-Winters best combo: {best_combo}, rmse_vali={best_rmse}")
        (best_trend, best_seasonal) = best_combo
        final_model = ExponentialSmoothing(
            train['unique_ships'],
            trend=best_trend,
            seasonal=best_seasonal,
            seasonal_periods=52
        )
        fit_final = final_model.fit()
        preds_test = fit_final.forecast(steps=len(test))
        # AIC e BIC (se supportato)
        try:
            aic = fit_final.aic
            bic = fit_final.bic
        except AttributeError:
            aic = np.nan
            bic = np.nan

        return preds_test.values, best_combo, aic, bic
    else:
        raise ValueError("Nessuna combo Holt-Winters valida trovata.")


# -------------------------------------------------------------------
# SCALING DI SUPPORTO
# -------------------------------------------------------------------

def scale_data(train_df, test_df, columns, scaler_type="minmax"):
    """
    Esegue lo scaling su 'columns' di train_df e test_df e restituisce i DF scalati e lo scaler.
    """
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(train_df[columns])
    train_df[columns] = scaler.transform(train_df[columns])
    test_df[columns] = scaler.transform(test_df[columns])
    return train_df, test_df, scaler


def scale_and_return(X_train, X_test, columns, scaler_type="minmax"):
    """
    Versione dedicata per ML: esegue lo scaling e restituisce X_train, X_test, scaler.
    """
    if scaler_type == "minmax":
        sc = MinMaxScaler()
    else:
        sc = StandardScaler()
    sc.fit(X_train[columns])
    X_train[columns] = sc.transform(X_train[columns])
    X_test[columns] = sc.transform(X_test[columns])
    return X_train, X_test, sc


# -------------------------------------------------------------------
# RANDOM FOREST (mini grid)
# -------------------------------------------------------------------

def train_random_forest_tuning(train, test, lag=4):
    """Esempio di mini grid search su n_estimators e max_depth."""
    from sklearn.model_selection import train_test_split

    combined = pd.concat([train, test], ignore_index=True)
    combined_features = create_features_lag(combined, lag=lag)

    train_cutoff = len(create_features_lag(train, lag=lag))
    train_df = combined_features.iloc[:train_cutoff].copy()
    test_df = combined_features.iloc[train_cutoff:].copy()

    X = train_df.drop(['date', 'unique_ships'], axis=1)
    y = train_df['unique_ships']

    X_train_sub, X_vali_sub, y_train_sub, y_vali_sub = train_test_split(X, y, test_size=0.2, shuffle=False)

    cols_to_scale = X.columns
    X_train_sub, X_vali_sub, scaler = scale_and_return(X_train_sub, X_vali_sub, cols_to_scale)

    from sklearn.ensemble import RandomForestRegressor

    n_estimators_list = [100, 200]
    max_depth_list = [None, 10]
    best_rmse = np.inf
    best_params = None

    for n_est in tqdm(n_estimators_list, desc="RF: n_estimators", leave=False):
        for md in tqdm(max_depth_list, desc="RF: max_depth", leave=False):
            rf = RandomForestRegressor(n_estimators=n_est, max_depth=md, random_state=42)
            rf.fit(X_train_sub, y_train_sub)

            X_vali_scaled = scaler.transform(X_vali_sub)
            preds_vali = rf.predict(X_vali_scaled)
            rmse_vali = np.sqrt(mean_squared_error(y_vali_sub, preds_vali))
            if rmse_vali < best_rmse:
                best_rmse = rmse_vali
                best_params = (n_est, md)

    if not best_params:
        raise ValueError("Nessun param valido trovato per Random Forest.")
    else:
        tqdm.write(f"Miglior RF params: {best_params}, vali_rmse={best_rmse}")

    best_n, best_d = best_params
    X_train_full = train_df.drop(['date', 'unique_ships'], axis=1)
    y_train_full = train_df['unique_ships']
    X_test_full = test_df.drop(['date', 'unique_ships'], axis=1)

    X_train_full, X_test_full, full_scaler = scale_and_return(X_train_full, X_test_full, cols_to_scale)

    final_rf = RandomForestRegressor(n_estimators=best_n, max_depth=best_d, random_state=42)
    final_rf.fit(X_train_full, y_train_full)
    preds_test = final_rf.predict(X_test_full)

    # AIC/BIC non applicabili
    aic = np.nan
    bic = np.nan

    return preds_test, aic, bic


# -------------------------------------------------------------------
# LSTM / GRU TUNING
# -------------------------------------------------------------------

def train_lstm_tuning(train, test, lag=4, epochs=20, batch_size=16, hidden_units=64):
    """Allena una LSTM con parametri base + dropout e restituisce le previsioni."""
    from sklearn.model_selection import train_test_split

    combined = pd.concat([train, test], ignore_index=True)
    combined_features = create_features_lag(combined, lag=lag)

    train_cutoff = len(create_features_lag(train, lag=lag))
    train_df = combined_features.iloc[:train_cutoff].copy()
    test_df = combined_features.iloc[train_cutoff:].copy()

    X = train_df.drop(['date', 'unique_ships'], axis=1).values
    y = train_df['unique_ships'].values

    X_train_sub, X_vali_sub, y_train_sub, y_vali_sub = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = MinMaxScaler()
    scaler.fit(X_train_sub)
    X_train_sub = scaler.transform(X_train_sub)
    X_vali_sub = scaler.transform(X_vali_sub)

    # Reshape per LSTM
    X_train_sub = X_train_sub.reshape((X_train_sub.shape[0], 1, X_train_sub.shape[1]))
    X_vali_sub = X_vali_sub.reshape((X_vali_sub.shape[0], 1, X_vali_sub.shape[1]))

    model = Sequential()
    model.add(LSTM(hidden_units, activation='relu', input_shape=(1, X_train_sub.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)

    tqdm.write("Inizio addestramento LSTM (mini-vali)...")
    model.fit(X_train_sub, y_train_sub,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_vali_sub, y_vali_sub),
              callbacks=[es, rlr],
              verbose=0)

    # Retrain su full train
    X_train_full = scaler.fit_transform(train_df.drop(['date', 'unique_ships'], axis=1).values)
    y_train_full = train_df['unique_ships'].values
    X_test_full = scaler.transform(test_df.drop(['date', 'unique_ships'], axis=1).values)

    X_train_full = X_train_full.reshape((X_train_full.shape[0], 1, X_train_full.shape[1]))
    X_test_full = X_test_full.reshape((X_test_full.shape[0], 1, X_test_full.shape[1]))

    final_model = Sequential()
    final_model.add(LSTM(hidden_units, activation='relu', input_shape=(1, X_train_full.shape[2])))
    final_model.add(Dropout(0.2))
    final_model.add(Dense(1))
    final_model.compile(optimizer='adam', loss='mse')

    tqdm.write("Retrain LSTM su tutto il train...")
    final_model.fit(X_train_full, y_train_full,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0)

    preds_test = final_model.predict(X_test_full).flatten()

    # AIC/BIC non definiti
    aic = np.nan
    bic = np.nan

    return preds_test, aic, bic


# -------------------------------------------------------------------
# SCRIPT PRINCIPALE
# -------------------------------------------------------------------

def main():
    # 1) Crea directory per salvare i risultati
    results_dir = create_unique_directory(
        base_path="results",
        prefix=f"prediction_{os.path.splitext(os.path.basename(sys.argv[0]))[0].split('_')[-1]}"
    )

    # 2) Caricamento e split
    tqdm.write("Caricamento e pre-elaborazione dati...")
    full_data = load_and_preprocess_data()
    train, test = split_data(full_data)
    ground_truth = test['unique_ships'].values

    # >>> Test stazionarietà ANCHE sulla serie originale <<<
    test_stationarity(full_data, col_name='unique_ships', results_dir=results_dir)

    # -------------------------------------------------------------
    # Plot base di TUTTO il dataset + differenze
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(full_data['date'], full_data['unique_ships'], label='Dataset completo')
    split_date = pd.to_datetime('2023-01-01')
    plt.axvline(split_date, color='r', linestyle='--', label='Train/Test split')
    plt.title('Dataset: Train + Test')
    plt.xlabel('Date')
    plt.ylabel('Unique Ships')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_full_dataset.png"))
    plt.close()

    # Plot differenziato 1 volta
    full_data['diff_1'] = full_data['unique_ships'].diff()
    plt.figure(figsize=(10, 5))
    plt.plot(full_data['date'], full_data['diff_1'], label='Differenziata 1 volta')
    plt.title('Dataset differenziato (1 volta)')
    plt.xlabel('Date')
    plt.ylabel('diff_1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_diff_1.png"))
    plt.close()

    # >>> TEST STAZIONARIETÀ su diff_1 <<<
    test_stationarity(full_data, col_name='diff_1', results_dir=results_dir)

    # Plot differenziato 2 volte
    full_data['diff_2'] = full_data['diff_1'].diff()
    plt.figure(figsize=(10, 5))
    plt.plot(full_data['date'], full_data['diff_2'], label='Differenziata 2 volte')
    plt.title('Dataset differenziato (2 volte)')
    plt.xlabel('Date')
    plt.ylabel('diff_2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "plot_diff_2.png"))
    plt.close()

    # >>> TEST STAZIONARIETÀ su diff_2 <<<
    test_stationarity(full_data, col_name='diff_2', results_dir=results_dir)

    all_metrics = []

    # ARIMA (param fissi)
    try:
        p_arima, d_arima, q_arima = 1, 1, 1
        arima_preds, arima_aic, arima_bic = train_arima(train, test, p_arima, d_arima, q_arima)
        arima_m = evaluate_metrics(ground_truth, arima_preds)
        arima_m["model"] = f"ARIMA_({p_arima},{d_arima},{q_arima})"
        arima_m["AIC"] = arima_aic
        arima_m["BIC"] = arima_bic
        all_metrics.append(arima_m)
        plot_prediction(train, test, arima_preds, arima_m["model"], results_dir)
    except Exception as e:
        tqdm.write(str(e))


    # SARIMAX
    try:
        p_smx, d_smx, q_smx = 1, 1, 1
        P_smx, D_smx, Q_smx, s_smx = 1, 1, 1, 52
        sarimax_preds, sarimax_aic, sarimax_bic = train_sarimax(train, test, p_smx, d_smx, q_smx, P_smx, D_smx, Q_smx, s_smx)
        sarimax_m = evaluate_metrics(ground_truth, sarimax_preds)
        sarimax_m["model"] = f"SARIMAX_({p_smx},{d_smx},{q_smx})x({P_smx},{D_smx},{Q_smx},{s_smx})"
        sarimax_m["AIC"] = sarimax_aic
        sarimax_m["BIC"] = sarimax_bic
        all_metrics.append(sarimax_m)
        plot_prediction(train, test, sarimax_preds, sarimax_m["model"], results_dir)
    except Exception as e:
        tqdm.write(str(e))

    # Holt-Winters
    tqdm.write("Holt-Winters con tuning (mini combos su trend/seasonal)...")
    try:
        hw_preds, hw_combo, hw_aic, hw_bic = train_holt_winters_tuning(train, test)
        hw_m = evaluate_metrics(ground_truth, hw_preds)
        hw_m["model"] = f"HoltWinters_{hw_combo}"
        hw_m["AIC"] = hw_aic
        hw_m["BIC"] = hw_bic
        all_metrics.append(hw_m)
        plot_prediction(train, test, hw_preds, hw_m["model"], results_dir)
    except Exception as e:
        tqdm.write(f"Holt-Winters Error: {str(e)}")

    # Random Forest
    tqdm.write("Random Forest con mini-grid (n_estimators, max_depth)...")
    try:
        rf_preds, rf_aic, rf_bic = train_random_forest_tuning(train, test, lag=4)
        rf_m = evaluate_metrics(ground_truth, rf_preds)
        rf_m["model"] = "RandomForest_tuned"
        rf_m["AIC"] = rf_aic
        rf_m["BIC"] = rf_bic
        all_metrics.append(rf_m)
        plot_prediction(train, test, rf_preds, "RandomForest_tuned", results_dir)
    except Exception as e:
        tqdm.write(f"RF Error: {str(e)}")

    # Gradient Boosting
    tqdm.write("Gradient Boosting param fissi (esempio)...")
    try:
        combined = pd.concat([train, test], ignore_index=True)
        combined_feat = create_features_lag(combined, lag=4)

        train_cutoff = len(create_features_lag(train, lag=4))
        train_df = combined_feat.iloc[:train_cutoff].copy()
        test_df = combined_feat.iloc[train_cutoff:].copy()

        X_train = train_df.drop(['date', 'unique_ships'], axis=1)
        y_train = train_df['unique_ships']
        X_test = test_df.drop(['date', 'unique_ships'], axis=1)

        X_train, X_test, sc_gb = scale_data(X_train, X_test, X_train.columns, scaler_type="minmax")

        gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
        gb_model.fit(X_train, y_train)
        gb_preds = gb_model.predict(X_test)

        gb_m = evaluate_metrics(ground_truth, gb_preds)
        gb_m["model"] = "GradientBoosting_tuned"
        gb_m["AIC"] = np.nan
        gb_m["BIC"] = np.nan
        all_metrics.append(gb_m)
        plot_prediction(train, test, gb_preds, "GradientBoosting_tuned", results_dir)
    except Exception as e:
        tqdm.write(f"GradientBoosting Error: {str(e)}")

    # LSTM
    tqdm.write("LSTM con param custom/tuning...")
    try:
        lstm_preds, lstm_aic, lstm_bic = train_lstm_tuning(train, test, lag=4, epochs=30, batch_size=16, hidden_units=64)
        lstm_m = evaluate_metrics(ground_truth, lstm_preds)
        lstm_m["model"] = "LSTM_tuned"
        lstm_m["AIC"] = lstm_aic
        lstm_m["BIC"] = lstm_bic
        all_metrics.append(lstm_m)
        plot_prediction(train, test, lstm_preds, "LSTM_tuned", results_dir)
    except Exception as e:
        tqdm.write(f"LSTM Error: {str(e)}")

    # GRU
    tqdm.write("GRU con param simili a LSTM (no grid in questo esempio)...")
    try:
        from sklearn.model_selection import train_test_split

        lag = 4
        combined = pd.concat([train, test], ignore_index=True)
        combined_feat = create_features_lag(combined, lag=lag)
        train_cutoff = len(create_features_lag(train, lag=lag))
        train_df = combined_feat.iloc[:train_cutoff].copy()
        test_df = combined_feat.iloc[train_cutoff:].copy()

        X = train_df.drop(['date', 'unique_ships'], axis=1).values
        y = train_df['unique_ships'].values
        X_train_sub, X_vali_sub, y_train_sub, y_vali_sub = train_test_split(X, y, test_size=0.2, shuffle=False)

        sc_gru = MinMaxScaler()
        sc_gru.fit(X_train_sub)
        X_train_sub = sc_gru.transform(X_train_sub)
        X_vali_sub = sc_gru.transform(X_vali_sub)

        X_train_sub = X_train_sub.reshape((X_train_sub.shape[0], 1, X_train_sub.shape[1]))
        X_vali_sub = X_vali_sub.reshape((X_vali_sub.shape[0], 1, X_vali_sub.shape[1]))

        model_gru = Sequential()
        model_gru.add(GRU(64, activation='relu', input_shape=(1, X_train_sub.shape[2])))
        model_gru.add(Dropout(0.2))
        model_gru.add(Dense(1))
        model_gru.compile(optimizer='adam', loss='mse')

        es_g = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        rlr_g = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
        tqdm.write("Addestramento GRU (mini-vali)...")
        model_gru.fit(X_train_sub, y_train_sub,
                      epochs=30, batch_size=16,
                      validation_data=(X_vali_sub, y_vali_sub),
                      callbacks=[es_g, rlr_g],
                      verbose=0)

        # Retrain su full train
        X_train_full = train_df.drop(['date', 'unique_ships'], axis=1).values
        y_train_full = train_df['unique_ships'].values
        X_test_full = test_df.drop(['date', 'unique_ships'], axis=1).values

        sc_gru.fit(X_train_full)
        X_train_full = sc_gru.transform(X_train_full)
        X_test_full = sc_gru.transform(X_test_full)

        X_train_full = X_train_full.reshape((X_train_full.shape[0], 1, X_train_full.shape[1]))
        X_test_full = X_test_full.reshape((X_test_full.shape[0], 1, X_test_full.shape[1]))

        tqdm.write("Retrain GRU su full train...")
        final_gru = Sequential()
        final_gru.add(GRU(64, activation='relu', input_shape=(1, X_train_full.shape[2])))
        final_gru.add(Dropout(0.2))
        final_gru.add(Dense(1))
        final_gru.compile(optimizer='adam', loss='mse')

        final_gru.fit(X_train_full, y_train_full, epochs=30, batch_size=16, verbose=0)

        gru_preds = final_gru.predict(X_test_full).flatten()
        gru_m = evaluate_metrics(ground_truth, gru_preds)
        gru_m["model"] = "GRU_tuned"
        gru_m["AIC"] = np.nan
        gru_m["BIC"] = np.nan
        all_metrics.append(gru_m)
        plot_prediction(train, test, gru_preds, "GRU_tuned", results_dir)

    except Exception as e:
        tqdm.write(f"GRU Error: {str(e)}")

    # Prophet
    tqdm.write("Prophet con param custom (seasonality_mode=multiplicative)...")
    try:
        df_train = train.rename(columns={'date': 'ds', 'unique_ships': 'y'}).copy()
        df_test = test.rename(columns={'date': 'ds', 'unique_ships': 'y'}).copy()

        model_prophet = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            seasonality_prior_scale=10
        )
        model_prophet.fit(df_train)

        future = model_prophet.make_future_dataframe(periods=len(test), freq='W')
        forecast = model_prophet.predict(future)
        forecast_test = forecast.tail(len(test))
        prophet_preds = forecast_test['yhat'].values

        prophet_m = evaluate_metrics(ground_truth, prophet_preds)
        prophet_m["model"] = "Prophet_multiplicative"
        prophet_m["AIC"] = np.nan
        prophet_m["BIC"] = np.nan
        all_metrics.append(prophet_m)
        plot_prediction(train, test, prophet_preds, "Prophet_multiplicative", results_dir)

    except Exception as e:
        tqdm.write(f"Prophet Error: {str(e)}")

    # -------------------------------------------------------------------
    # SALVATAGGIO RISULTATI FINALI
    # -------------------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df[['model', 'MAE', 'RMSE', 'MAPE', 'AIC', 'BIC']]
    out_csv = os.path.join(results_dir, "evaluation_metrics_tuned.csv")
    metrics_df.to_csv(out_csv, index=False)

    tqdm.write("==== RISULTATI FINALI ====")
    tqdm.write(str(metrics_df))
    tqdm.write(f"Salvato in: {out_csv}")
    tqdm.write("Script completato con successo!")


if __name__ == "__main__":
    main()
