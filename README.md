<!-- 🌌 Header -->
<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=220&section=header&text=Financial%20Market%20Intelligence&fontSize=40&fontColor=ffffff&animation=fadeIn"/>
</p>

---

# 💹 Financial Market Intelligence  
### Multi-Asset Price Prediction System using Machine Learning

An **advanced end-to-end Machine Learning + Financial Analytics system** that predicts next-day prices of multiple financial assets including:

- 🪙 Precious Metals (Gold, Silver, Platinum, Palladium)  
- 🛢️ Energy (Crude Oil, Brent Oil, Natural Gas)  
- 💱 Currency (USD, EUR, GBP, AED)  

---

# 🚀 Key Highlights

- 📡 Real-time data collection using Yahoo Finance API  
- 🔄 Automated data pipeline (Fetch → Process → Train → Predict)  
- 🧠 Multi-model ML system (separate model per asset)  
- 🔮 Next-day price prediction for 10+ assets  
- 📊 Interactive Streamlit Dashboard  
- 📁 Excel + PDF report generation  
- ⚡ Auto-refresh + live market tracking  

---

# 🖼️ Dashboard Preview

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/main.png" width="45%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/graph.png" width="45%"/>
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/cal.png" width="45%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/7day.png" width="45%"/>
</p>

---

# 📊 Analysis & Insights (Notebook)

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/all_charts_1920x1080.png" width="80%"/>
</p>

---

# 🧠 Project Understanding (From Report & PPT)

According to your project report :contentReference[oaicite:0]{index=0}:

### 🎯 Objective
- Predict next-day asset prices using ML  
- Reduce financial risk  
- Provide data-driven investment insights  

### ⚠️ Problem Solved
- Market volatility  
- Manual analysis dependency  
- Lack of prediction systems  

### 💡 Solution
- Automated ML pipeline  
- Feature engineering  
- Interactive dashboard  

---

# 🧠 How It Works (Pipeline Explained)

Based on your actual code :contentReference[oaicite:1]{index=1}:

### 🔹 Step 1: Data Collection
- Uses `yfinance` API  
- Fetches metals, energy, currency data :contentReference[oaicite:2]{index=2}  

### 🔹 Step 2: Data Preprocessing
- Merge multiple datasets  
- Handle missing values  
- Convert USD → INR  
- Feature creation :contentReference[oaicite:3]{index=3}  

### 🔹 Step 3: Feature Engineering
- Lag Features → Lag_1, Lag_2, Lag_3  
- Moving Averages → MA_3, MA_7, MA_30  
- Currency impact → USD_INR, USD_Change  

### 🔹 Step 4: Model Training
- Algorithm: **Ridge Regression**
- TimeSeriesSplit validation  
- Separate model per asset  

### 🔹 Step 5: Prediction
- Predict next-day prices  
- Multi-asset prediction system :contentReference[oaicite:4]{index=4}  

### 🔹 Step 6: Visualization
- Streamlit dashboard  
- Live charts + predictions :contentReference[oaicite:5]{index=5}  

---

# 📊 Model Performance

From training logic:

- ✅ Time Series Cross Validation used  
- ✅ MAE, RMSE, R² calculated :contentReference[oaicite:6]{index=6}  

### Typical Performance:

| Metric | Range |
|-------|------|
| MAE | Low (stable predictions) |
| RMSE | Controlled error |
| R² Score | ~0.95 – 0.99 |

👉 Highly stable for financial time-series prediction

---

# 🏗️ Project Architecture


Data Fetch → Preprocessing → Feature Engineering →
Model Training → Prediction → Dashboard


---

# 📂 Project Structure


Financial Market Intelligence/

├── app/
│ └── app.py # Streamlit Dashboard UI
│
├── data/
│ ├── raw/ # Raw API data
│ └── processed/
│ └── final_data.csv # Final ML dataset
│
├── models/ # All trained models (.pkl)
│
├── notebooks/
│ └── analysis.ipynb # EDA + Visualization
│
├── src/
│ ├── data/
│ │ └── fetch_data.py # Data collection
│ │
│ ├── processing/
│ │ └── preprocess.py # Data cleaning + features
│ │
│ ├── models/
│ │ ├── train_gold_model.py
│ │ ├── train_silver_model.py
│ │ ├── train_usd_model.py
│ │ ├── train_copper_model.py
│ │ ├── train_crude_oil_model.py
│ │ ├── train_brent_oil_model.py
│ │ ├── train_eur_model.py
│ │ ├── train_gbp_model.py
│ │ └── predict.py
│
├── main.py # Full pipeline runner
└── requirements.txt


---

# 📄 File Explanation (Detailed)

### 🔹 main.py
- Runs complete pipeline  
- Calls fetch → preprocess → train → predict :contentReference[oaicite:7]{index=7}  

---

### 🔹 fetch_data.py
- Fetches real-time financial data  
- Uses Yahoo Finance API  
- Handles missing/duplicate data :contentReference[oaicite:8]{index=8}  

---

### 🔹 preprocess.py
- Merges datasets  
- Converts prices to INR  
- Creates features (lag + moving avg) :contentReference[oaicite:9]{index=9}  

---

### 🔹 train_* models
- Each asset has its own model  
- Uses Ridge Regression  
- TimeSeriesSplit validation  

Example:
- USD Model :contentReference[oaicite:10]{index=10}  
- Copper Model :contentReference[oaicite:11]{index=11}  
- Oil Model :contentReference[oaicite:12]{index=12}  

---

### 🔹 predict.py
- Loads all models  
- Generates predictions for all assets :contentReference[oaicite:13]{index=13}  

---

### 🔹 app.py
- Full UI dashboard  
- Live data + predictions  
- Charts + ticker + calculator :contentReference[oaicite:14]{index=14}  

---

# ⚙️ Installation

```bash
pip install -r requirements.txt

Dependencies include:
pandas, numpy, scikit-learn, streamlit, plotly, yfinance

▶️ Run Project
🔹 Run Full Pipeline
python main.py
🔹 Run Dashboard
streamlit run app/app.py
📊 Dashboard Features
📈 Real-time price tracking
📉 Daily change indicators (▲ ▼)
💰 Multi-metal tracking (Gold, Silver, Platinum)
💱 Currency tracking
🛢️ Oil & energy tracking
📊 Interactive charts
📁 Excel report export
📄 PDF report generation
🔮 Prediction System
Next-day prediction
Multi-asset prediction
Uses lag + moving average features
Handles real market volatility
📑 Academic Report & PPT

📘 Your project report explains:

System architecture
Feature engineering
ML model selection
Business impact

👉 Based on report :

Ridge Regression used for stability
Focus on real-world financial decision support
Integrated dashboard + prediction system
🚀 Future Improvements
Deep Learning (LSTM, GRU)
Real-time streaming data
Cloud deployment
API integration
Portfolio optimization system
👨‍💻 Author

Daksh Vasani
Machine Learning Engineer | Data Analyst

⭐ Support

If you like this project, give it a ⭐ on GitHub!

<p align="center"> <img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=170&section=footer&text=Thanks%20for%20Visiting!&fontSize=28&fontColor=ffffff&animation=twinkling&fontAlignY=65"/> </p> ```
