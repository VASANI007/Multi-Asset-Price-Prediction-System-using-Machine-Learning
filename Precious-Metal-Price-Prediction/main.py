import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback

# DATA
from src.data.fetch_data import fetch_all
from src.processing.preprocess import preprocess

# MODELS
from src.models.train_gold_model import train_model as train_gold_model
from src.models.train_silver_model import train_silver_model
from src.models.train_usd_model import train_usd_model
from src.models.train_eur_model import train_eur_model
from src.models.train_gbp_model import train_gbp_model

from src.models.train_platinum_model import train_platinum_model
from src.models.train_palladium_model import train_palladium_model
from src.models.train_copper_model import train_copper_model

from src.models.train_crude_oil_model import train_crude_oil_model
from src.models.train_brent_oil_model import train_brent_oil_model
from src.models.train_natural_gas_model import train_natural_gas_model

# PREDICTION
from src.models.predict import predict_all


def run_pipeline():

    print("\n🚀 Starting Full Multi-Asset Pipeline...\n")

    try:

        # ---------------- STEP 1 ----------------
        print("📡 Fetching Data...")
        fetch_all()
        print("✅ Data Fetch Complete\n")

        # ---------------- STEP 2 ----------------
        print("⚙️ Processing Data...")
        preprocess()
        print("✅ Data Processing Complete\n")

        # ---------------- STEP 3 ----------------
        print("🧠 Training Models...\n")

        train_gold_model()
        train_silver_model()
        train_usd_model()

        train_eur_model()
        train_gbp_model()

        train_platinum_model()
        train_palladium_model()
        train_copper_model()

        train_crude_oil_model()
        train_brent_oil_model()
        train_natural_gas_model()

        print("\n✅ All Models Trained Successfully\n")

        # ---------------- STEP 4 ----------------
        print("🔮 Running Predictions...\n")

        results = predict_all()

        print("📊 Latest Predictions:\n")

        for name, value in results.items():
            print(f"{name}: ₹ {value:.2f}")

        print("\n✅ Prediction Complete\n")

    except Exception as e:

        print("\n❌ Pipeline Failed!")
        print(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    run_pipeline()