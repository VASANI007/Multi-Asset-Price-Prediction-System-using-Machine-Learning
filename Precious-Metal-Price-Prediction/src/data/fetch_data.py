import yfinance as yf
import pandas as pd
import os


# FILE PATHS


DATA_DIR = "data/raw"
os.makedirs(DATA_DIR, exist_ok=True)


# SYMBOL MAP (METALS + ENERGY + CURRENCY 🔥)


SYMBOLS = {
    # Metals
    "gold": "GC=F",
    "silver": "SI=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "copper": "HG=F",
    "aluminum": "ALI=F",

    # Energy
    "crude_oil": "CL=F",
    "brent_oil": "BZ=F",
    "natural_gas": "NG=F",

    # 💱 CURRENCIES (IMPORTANT ADD)
    "usd_inr": "USDINR=X",
    "eur_inr": "EURINR=X",
    "gbp_inr": "GBPINR=X",
    "aed_inr": "AEDINR=X",
    "jpy_inr": "JPYINR=X",

    # Global pairs (optional but powerful)
    "eur_usd": "EURUSD=X",
    "gbp_usd": "GBPUSD=X"
}


# SAFE DOWNLOAD FUNCTION

def safe_download(symbol):
    df = yf.download(symbol, start="2015-01-01", interval="1d", progress=False)

    if df.empty:
        raise ValueError(f"No data fetched for {symbol}")

    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    # Ensure Date column
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    return df


# FETCH FUNCTION
def fetch_data(name, symbol):
    file_path = os.path.join(DATA_DIR, f"{name}_raw.csv")

    print(f"\nFetching {name.upper()} ({symbol})...")

    try:
        new_data = safe_download(symbol)
        print("New data fetched")
        print("Latest new date:", new_data['Date'].max().date())

    except Exception as e:
        print(f"Fetch failed: {e}")
        return pd.DataFrame()

    # Merge with old data
    if os.path.exists(file_path):
        try:
            old_data = pd.read_csv(file_path, low_memory=False)

            if not old_data.empty:
                old_data['Date'] = pd.to_datetime(old_data['Date'], errors='coerce')
                old_data = old_data.dropna(subset=['Date'])

                df = pd.concat([old_data, new_data])
                df = df.drop_duplicates(subset='Date')
                df = df.sort_values('Date').reset_index(drop=True)
            else:
                df = new_data

        except Exception as e:
            print(f"Old file corrupted → rebuilding: {e}")
            df = new_data
    else:
        print("First-time download")
        df = new_data

    if df.empty:
        raise ValueError(f"Final dataset empty for {symbol}")

    # Save file
    df.to_csv(file_path, index=False)

    print(f"Saved → {file_path}")
    print(f"Final latest date → {df['Date'].max().date()}")

    return df


# FETCH ALL
def fetch_all():
    print("\nFetching All Market Data...\n")

    results = {}

    for name, symbol in SYMBOLS.items():
        df = fetch_data(name, symbol)
        results[name] = df

    print("\nAll Data Updated!\n")

    return results


# MAIN
if __name__ == "__main__":
    print("Starting Data Fetch...\n")
    fetch_all()
    print("\nDone!")