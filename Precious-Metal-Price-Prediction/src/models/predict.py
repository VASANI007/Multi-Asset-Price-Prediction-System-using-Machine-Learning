import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import pickle


DATA_FILE = "data/processed/final_data.csv"

MODEL_PATHS = {
    "gold": "models/gold_model.pkl",
    "silver": "models/silver_model.pkl",
    "usd": "models/usd_model.pkl",
    "eur": "models/eur_model.pkl",
    "gbp": "models/gbp_model.pkl",
    "platinum": "models/platinum_model.pkl",
    "palladium": "models/palladium_model.pkl",
    "copper": "models/copper_model.pkl",
    "crude_oil": "models/crude_oil_model.pkl",
    "brent_oil": "models/brent_oil_model.pkl",
    "natural_gas": "models/natural_gas_model.pkl"
}


def load_data():
    df = pd.read_csv(DATA_FILE)
    df['Date'] = pd.to_datetime(df['Date'])

    df = df[df['Date'] >= df['Date'].max() - pd.DateOffset(years=1)]

    return df


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------- GOLD ----------------
def predict_gold(df, model):
    lag1 = df['Gold_24K_1g'].iloc[-1]
    lag2 = df['Gold_24K_1g'].iloc[-2]
    lag3 = df['Gold_24K_1g'].iloc[-3]

    ma7 = df['Gold_24K_1g'].tail(7).mean()
    ma30 = df['Gold_24K_1g'].tail(30).mean()

    usd = df['USD_INR'].iloc[-1]
    usd_change = df['USD_INR'].pct_change().iloc[-1]

    gold22 = df['Gold_22K_1g'].iloc[-1]
    gold18 = df['Gold_18K_1g'].iloc[-1]

    ratio = gold18 / lag1 if lag1 != 0 else 0
    day = df['Date'].iloc[-1].dayofweek

    X = pd.DataFrame([[ 
        lag1, lag2, lag3,
        ma7, ma30,
        usd, usd_change,
        gold22, gold18,
        ratio,
        day
    ]], columns=[
        'Lag_1','Lag_2','Lag_3',
        'MA_7','MA_30',
        'USD_INR','USD_Change',
        'Gold_22K_1g','Gold_18K_1g',
        'Gold_18K_Ratio',
        'DayOfWeek'
    ])

    return model.predict(X)[0]


# ---------------- GENERIC ----------------
def create_features(series, usd=None):
    lag1 = series.iloc[-1]
    lag2 = series.iloc[-2]
    lag3 = series.iloc[-3]

    ma3 = series.tail(3).mean()
    ma7 = series.tail(7).mean()

    data = [lag1, lag2, lag3, ma3, ma7]
    cols = ['Lag_1','Lag_2','Lag_3','MA_3','MA_7']

    if usd is not None:
        usd_val = usd.iloc[-1]
        usd_change = usd.pct_change().iloc[-1]
        data += [usd_val, usd_change]
        cols += ['USD_INR','USD_Change']

    return pd.DataFrame([data], columns=cols)


# ---------------- OTHER MODELS ----------------
def predict_silver(df, model):
    return model.predict(create_features(df['Silver_1g'], df['USD_INR']))[0]

def predict_usd(df, model):
    return model.predict(create_features(df['USD_INR']))[0]

def predict_eur(df, model):
    return model.predict(create_features(df['EUR_INR']))[0]

def predict_gbp(df, model):
    return model.predict(create_features(df['GBP_INR']))[0]

def predict_platinum(df, model):
    return model.predict(create_features(df['Platinum_1g'], df['USD_INR']))[0]

def predict_palladium(df, model):
    return model.predict(create_features(df['Palladium_1g'], df['USD_INR']))[0]

def predict_copper(df, model):
    return model.predict(create_features(df['Copper_1g'], df['USD_INR']))[0]

def predict_crude_oil(df, model):
    return model.predict(create_features(df['Crude_Oil_INR_per_barrel'], df['USD_INR']))[0]

def predict_brent_oil(df, model):
    return model.predict(create_features(df['Brent_Oil_INR_per_barrel'], df['USD_INR']))[0]

def predict_natural_gas(df, model):
    return model.predict(create_features(df['Natural_Gas_INR'], df['USD_INR']))[0]


# ---------------- CURRENT PREDICTION ----------------
def predict_all():

    df = load_data()
    models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

    return {
        "Gold": predict_gold(df, models["gold"]),
        "Silver": predict_silver(df, models["silver"]),
        "USD": predict_usd(df, models["usd"]),
        "EUR": predict_eur(df, models["eur"]),
        "GBP": predict_gbp(df, models["gbp"]),
        "Platinum": predict_platinum(df, models["platinum"]),
        "Palladium": predict_palladium(df, models["palladium"]),
        "Copper": predict_copper(df, models["copper"]),
        "Crude Oil": predict_crude_oil(df, models["crude_oil"]),
        "Brent Oil": predict_brent_oil(df, models["brent_oil"]),
        "Natural Gas": predict_natural_gas(df, models["natural_gas"])
    }


# ---------------- FUTURE PREDICTION ----------------
def predict_future_all(days=7):

    df = load_data()
    models = {name: load_model(path) for name, path in MODEL_PATHS.items()}

    future_results = {name: [] for name in MODEL_PATHS.keys()}
    temp_df = df.copy()

    for _ in range(days):

        gold = predict_gold(temp_df, models["gold"])
        silver = predict_silver(temp_df, models["silver"])
        usd = predict_usd(temp_df, models["usd"])
        eur = predict_eur(temp_df, models["eur"])
        gbp = predict_gbp(temp_df, models["gbp"])
        platinum = predict_platinum(temp_df, models["platinum"])
        palladium = predict_palladium(temp_df, models["palladium"])
        copper = predict_copper(temp_df, models["copper"])
        crude = predict_crude_oil(temp_df, models["crude_oil"])
        brent = predict_brent_oil(temp_df, models["brent_oil"])
        gas = predict_natural_gas(temp_df, models["natural_gas"])

        future_results["gold"].append(round(gold, 2))
        future_results["silver"].append(round(silver, 2))
        future_results["usd"].append(round(usd, 2))
        future_results["eur"].append(round(eur, 2))
        future_results["gbp"].append(round(gbp, 2))
        future_results["platinum"].append(round(platinum, 2))
        future_results["palladium"].append(round(palladium, 2))
        future_results["copper"].append(round(copper, 2))
        future_results["crude_oil"].append(round(crude, 2))
        future_results["brent_oil"].append(round(brent, 2))
        future_results["natural_gas"].append(round(gas, 2))

        new_row = temp_df.iloc[-1].copy()
        new_row['Date'] += pd.Timedelta(days=1)

        new_row['Gold_24K_1g'] = gold
        new_row['Silver_1g'] = silver
        new_row['USD_INR'] = usd
        new_row['EUR_INR'] = eur
        new_row['GBP_INR'] = gbp
        new_row['Platinum_1g'] = platinum
        new_row['Palladium_1g'] = palladium
        new_row['Copper_1g'] = copper
        new_row['Crude_Oil_INR_per_barrel'] = crude
        new_row['Brent_Oil_INR_per_barrel'] = brent
        new_row['Natural_Gas_INR'] = gas

        temp_df = pd.concat([temp_df, pd.DataFrame([new_row])], ignore_index=True)

    return future_results


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("Running Full Multi-Asset Prediction...\n")

    preds = predict_all()

    for k, v in preds.items():
        print(f"{k}: ₹ {v:.2f}")

    print("\nNext 7 Days Forecast:\n")

    future = predict_future_all(7)

    for i in range(7):
        print(f"\nDay {i+1}:")
        for k in future:
            print(f"{k}: ₹ {future[k][i]}")

    print("\nDone")