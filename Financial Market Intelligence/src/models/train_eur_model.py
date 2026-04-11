import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import pickle
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# file paths
DATA_FILE = "data/processed/final_data.csv"
MODEL_FILE = "models/eur_model.pkl"
METRICS_FILE = "models/eur_metrics.pkl"


def train_eur_model():

    # load dataset
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])

    # keep only EUR data
    df = df[['Date', 'EUR_INR']].dropna()

    # lag features
    df['Lag_1'] = df['EUR_INR'].shift(1)
    df['Lag_2'] = df['EUR_INR'].shift(2)
    df['Lag_3'] = df['EUR_INR'].shift(3)

    # moving averages
    df['MA_3'] = df['EUR_INR'].rolling(3).mean()
    df['MA_7'] = df['EUR_INR'].rolling(7).mean()

    # target = next day EUR value
    df['Target'] = df['EUR_INR'].shift(-1)

    # clean data
    df = df.dropna()

    # features and labels
    X = df[['Lag_1','Lag_2','Lag_3','MA_3','MA_7']]
    y = df['Target']

    # model
    model = Ridge(alpha=0.5)

    # cross validation
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

    # final metrics
    final_mae = float(np.mean(mae_list))
    final_r2 = float(np.mean(r2_list))
    final_rmse = float(np.mean(rmse_list))

    print("\n EUR Model Performance:")
    print(f"MAE: {final_mae:.2f}")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"RMSE: {final_rmse:.2f}")

    # save metrics
    metrics = {
        "MAE": final_mae,
        "RMSE": final_rmse,
        "R2": final_r2
    }

    os.makedirs("models", exist_ok=True)

    with open(METRICS_FILE, "wb") as f:
        pickle.dump(metrics, f)

    print(" Metrics saved -> models/eur_metrics.pkl")

    # final training
    model.fit(X, y)

    # save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print(" EUR model saved -> models/eur_model.pkl")

    return model


if __name__ == "__main__":
    print("Training EUR Model...\n")
    train_eur_model()
    print("\n All Done!")