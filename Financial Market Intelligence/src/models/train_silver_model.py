import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import pickle
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

def train_silver_model():

    
    # LOAD DATA
    df = pd.read_csv("data/processed/final_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])

    
    # SELECT ONLY SILVER DATA
    df = df[
        [
            'Date',
            'Silver_1g',
            'USD_INR'
        ]
    ].dropna()

    
    # FEATURE ENGINEERING
    df['Lag_1'] = df['Silver_1g'].shift(1)
    df['Lag_2'] = df['Silver_1g'].shift(2)
    df['Lag_3'] = df['Silver_1g'].shift(3)

    df['MA_3'] = df['Silver_1g'].rolling(3).mean()
    df['MA_7'] = df['Silver_1g'].rolling(7).mean()

    # USD influence
    df['USD_Change'] = df['USD_INR'].pct_change()

    # TARGET (next day)
    df['Target'] = df['Silver_1g'].shift(-1)

    df = df.dropna()
    # FEATURES & LABEL
    X = df[
        [
            'Lag_1','Lag_2','Lag_3',
            'MA_3','MA_7',
            'USD_INR',
            'USD_Change'
        ]
    ]

    y = df['Target']

    
    # MODEL
    model = Ridge(alpha=0.5)

    
    # CROSS VALIDATION
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

    
    # FINAL METRICS
    final_mae = float(np.mean(mae_list))
    final_r2 = float(np.mean(r2_list))
    final_rmse = float(np.mean(rmse_list))

    print("\n Silver Model Performance:")
    print(f"MAE: {final_mae:.2f}")
    print(f"R2 Score: {final_r2:.4f}")
    print(f"RMSE: {final_rmse:.2f}")

    
    # SAVE METRICS
    metrics = {
        "MAE": final_mae,
        "RMSE": final_rmse,
        "R2": final_r2
    }

    os.makedirs("models", exist_ok=True)

    with open("models/silver_metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print(" Metrics saved -> models/silver_metrics.pkl")

    
    # FINAL TRAIN
    model.fit(X, y)

    with open("models/silver_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(" Silver model saved -> models/silver_model.pkl")

    return model

# MAIN
if __name__ == "__main__":
    print("Training Silver Model...\n")
    train_silver_model()
    print("\n All Done!")