import pandas as pd
import yfinance as yf
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# FILE PATHS
DATA_DIR = "data/raw"
OUTPUT_FILE = "data/processed/final_data.csv"

FILES = {
    "gold": "gold_raw.csv",
    "silver": "silver_raw.csv",
    "platinum": "platinum_raw.csv",
    "palladium": "palladium_raw.csv",
    "copper": "copper_raw.csv",
    "crude_oil": "crude_oil_raw.csv",
    "brent_oil": "brent_oil_raw.csv",
    "natural_gas": "natural_gas_raw.csv",

    # currencies added
    "usd_inr": "usd_inr_raw.csv",
    "eur_inr": "eur_inr_raw.csv",
    "gbp_inr": "gbp_inr_raw.csv",
    "aed_inr": "aed_inr_raw.csv"
}

OUNCE_TO_GRAM = 31.1035
POUND_TO_GRAM = 453.592


# DUTY FACTORS
DUTY = {
    "gold": 1.08,
    "silver": 1.06,
    "platinum": 1.10,
    "palladium": 1.10,
    "copper": 1.04,
    "crude_oil": 1.03,
    "brent_oil": 1.03,
    "natural_gas": 1.02
}


# LOAD DATA
def load_all_data():
    data = {}

    for name, file in FILES.items():
        path = os.path.join(DATA_DIR, file)

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'Close'])

        df = df[['Date', 'Close']]
        df.rename(columns={'Close': name.upper()}, inplace=True)

        data[name] = df

    return data


# PREPROCESS
def preprocess():

    data = load_all_data()

    df = None
    for d in data.values():
        if df is None:
            df = d
        else:
            df = pd.merge(df, d, on="Date", how="outer")

    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df = df.resample('D').ffill()

    df['Date'] = df.index
    df.reset_index(drop=True, inplace=True)

    df = df.ffill().bfill()

    
    # CURRENCY HANDLING (CORE)
    df['USD_INR'] = df['USD_INR']
    df['EUR_INR'] = df.get('EUR_INR', df['USD_INR'] * 0.92)
    df['GBP_INR'] = df.get('GBP_INR', df['USD_INR'] * 0.78)

    
    # METALS (GRAM BASED)
    df['Gold_24K_1g'] = (df['GOLD'] / OUNCE_TO_GRAM) * df['USD_INR'] * DUTY['gold']
    df['Silver_1g'] = (df['SILVER'] / OUNCE_TO_GRAM) * df['USD_INR'] * DUTY['silver']

    df['Gold_22K_1g'] = df['Gold_24K_1g'] * (22/24)
    df['Gold_18K_1g'] = df['Gold_24K_1g'] * (18/24)

    # Platinum / Palladium (ounce → gram)
    for metal in ["platinum", "palladium"]:
        col = metal.upper()
        if col in df.columns:
            df[f"{metal.capitalize()}_1g"] = (df[col] / OUNCE_TO_GRAM) * df['USD_INR'] * DUTY[metal]

    # Copper (pound → gram)
    if 'COPPER' in df.columns:
        df['Copper_1g'] = (df['COPPER'] / POUND_TO_GRAM) * df['USD_INR'] * DUTY['copper']

    
    # ENERGY (KEEP UNIT SEPARATE)
    if 'CRUDE_OIL' in df.columns:
        df['Crude_Oil_INR_per_barrel'] = df['CRUDE_OIL'] * df['USD_INR'] * DUTY['crude_oil']

    if 'BRENT_OIL' in df.columns:
        df['Brent_Oil_INR_per_barrel'] = df['BRENT_OIL'] * df['USD_INR'] * DUTY['brent_oil']

    if 'NATURAL_GAS' in df.columns:
        df['Natural_Gas_INR'] = df['NATURAL_GAS'] * df['USD_INR'] * DUTY['natural_gas']

    
    # FEATURES
    df['Gold_Return'] = df['Gold_24K_1g'].pct_change()
    df['Silver_Return'] = df['Silver_1g'].pct_change()

    df['Gold_Volatility'] = df['Gold_Return'].rolling(7).std()

    df['Gold_Silver_Ratio'] = df['Gold_24K_1g'] / df['Silver_1g']

    
    # WEIGHTS
    weights = {"1g":1, "10g":10, "100g":100, "1kg":1000}

    for metal in ["Gold_24K", "Gold_22K", "Gold_18K", "Silver"]:
        for w, val in weights.items():
            df[f"{metal}_{w}"] = df[f"{metal}_1g"] * val

    if 'Platinum_1g' in df.columns:
        for w, val in weights.items():
            df[f"Platinum_{w}"] = df['Platinum_1g'] * val

    
    # FINAL CLEAN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=['Gold_24K_1g', 'Silver_1g'])

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nFinal data saved →", OUTPUT_FILE)

    return df

if __name__ == "__main__":
    print("Processing Data...\n")
    preprocess()
    print("\nDone")