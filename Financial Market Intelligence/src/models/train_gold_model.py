import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


DATA_FILE = "data/processed/final_data.csv"
MODEL_FILE = "models/gold_model.pkl"
METRICS_FILE = "models/gold_metrics.pkl"


def load_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def prepare_features(df):

    df = df[
        [
            'Date',
            'Gold_24K_1g',
            'Gold_22K_1g',
            'Gold_18K_1g',
            'USD_INR'
        ]
    ].dropna()

    df['Lag_1'] = df['Gold_24K_1g'].shift(1)
    df['Lag_2'] = df['Gold_24K_1g'].shift(2)
    df['Lag_3'] = df['Gold_24K_1g'].shift(3)

    df['MA_7'] = df['Gold_24K_1g'].rolling(7).mean()
    df['MA_30'] = df['Gold_24K_1g'].rolling(30).mean()

    df['USD_Change'] = df['USD_INR'].pct_change()

    df['Gold_18K_Ratio'] = df['Gold_18K_1g'] / df['Gold_24K_1g']

    df['DayOfWeek'] = df['Date'].dt.dayofweek

    df = df.dropna()
    return df


def train_model():

    df = load_data()
    df = prepare_features(df)

    df['Target'] = df['Gold_24K_1g'].shift(-1)
    df = df.dropna()

    X = df[
        [
            'Lag_1', 'Lag_2', 'Lag_3',
            'MA_7', 'MA_30',
            'USD_INR',
            'USD_Change',
            'Gold_22K_1g',
            'Gold_18K_1g',
            'Gold_18K_Ratio',
            'DayOfWeek'
        ]
    ]

    y = df['Target']

    model = Ridge(alpha=0.5)

    tscv = TimeSeriesSplit(n_splits=5)

    mae_list, r2_list, rmse_list = [], [], []

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, y_pred))
        r2_list.append(r2_score(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))

    final_mae = float(np.mean(mae_list))
    final_r2 = float(np.mean(r2_list))
    final_rmse = float(np.mean(rmse_list))

    print("\n Gold Model Performance:")
    print(f"MAE: {final_mae:.2f}")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"RMSE: {final_rmse:.2f}")

    metrics = {
        "MAE": final_mae,
        "RMSE": final_rmse,
        "R2": final_r2
    }

    os.makedirs("models", exist_ok=True)

    with open(METRICS_FILE, "wb") as f:
        pickle.dump(metrics, f)

    print(" Metrics saved -> models/gold_metrics.pkl")

    model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(" Model saved -> models/gold_model.pkl")

    return model


if __name__ == "__main__":
    print("Training Gold Model...\n")
    train_model()
    print("\n All Done!")