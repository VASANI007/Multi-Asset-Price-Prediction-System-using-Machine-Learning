<!-- 🌌 Header -->
<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=220&section=header&text=Financial%20Market%20Intelligence&fontSize=40&fontColor=ffffff&animation=fadeIn"/>
</p>

---

# 💹 Financial Market Intelligence  
### Multi-Asset Price Prediction System using Machine Learning

An **advanced end-to-end Machine Learning + Financial Analytics system** that predicts next-day prices of multiple financial assets.

---

# 🚀 Key Highlights

- 📡 Real-time data collection using Yahoo Finance API  
- 🔄 Automated ML pipeline (Fetch → Process → Train → Predict)  
- 🧠 Multi-model system (separate model per asset)  
- 🔮 Next-day prediction for multiple assets  
- 📊 Interactive Streamlit Dashboard  
- 📁 Excel + PDF reports  
- ⚡ Auto-refresh + live tracking  

---

# 🎯 Problem Statement

Financial markets are:
- Highly volatile 📉  
- Hard to predict manually 🤯  
- Time-consuming ⏳  

👉 This project solves:
- Manual analysis dependency  
- Lack of prediction tools  
- Poor decision-making support  

---

# 💡 Solution

✔ Automated pipeline  
✔ Machine Learning predictions  
✔ Multi-asset analysis  
✔ Interactive dashboard  

---

# 🖼️ Dashboard Preview

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/main.png" width="24%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/7day.png" width="24%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/graph.png" width="24%"/>
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/cal.png" width="24%"/>
</p>

### 🔍 What Each Image Shows

- **main.png** → Main dashboard UI  
- **graph.png** → Price trends  
- **cal.png** → Calculator  
- **7day.png** → Prediction view  

---

# 📊 Analysis (Notebook)

<p align="center">
<img src="https://raw.githubusercontent.com/VASANI007/Multi-Asset-Price-Prediction-System-using-Machine-Learning/main/Financial%20Market%20Intelligence/images/all_charts_1920x1080.png" width="85%"/>
</p>

---

# 🧠 How It Works (Deep Explanation)

## 🔹 Step 1: Data Collection
- Uses `yfinance` API  
- Collects:
  - Gold, Silver  
  - Oil prices  
  - Currency exchange  

---

## 🔹 Step 2: Data Preprocessing
- Merge datasets  
- Handle missing values  
- Convert USD → INR  
- Clean data  

---

## 🔹 Step 3: Feature Engineering

Features created:

- Lag Features → Lag_1, Lag_2  
- Moving Averages → MA_3, MA_7  
- Currency effect → USD_INR  

👉 This step improves accuracy significantly  

---

## 🔹 Step 4: Model Training

- Algorithm: **Ridge Regression**
- Why:
  - Stable  
  - Handles multicollinearity  

- Validation:
  - TimeSeriesSplit  

---

## 🔹 Step 5: Prediction

- Predict next-day price  
- Uses trained models  
- Works for multiple assets  

---

## 🔹 Step 6: Visualization

- Built using Streamlit  
- Shows:
  - Charts  
  - Predictions  
  - Insights  

---

# 📊 Model Performance

| Metric | Value |
|-------|------|
| MAE | Low |
| RMSE | Stable |
| R² Score | 0.95 – 0.99 |

👉 High accuracy for financial prediction  

---

# 🏗️ Architecture

Data Fetch → Preprocess → Feature Engineering → Train → Predict → Dashboard  

---

# 📂 Project Structure

```
Financial Market Intelligence/

├── app/
│   └── app.py
│
├── data/
│   ├── raw/ # all data.csv file
│   └── processed/ #final_data.csv
│
├── models/ # all train models.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── src/
│   ├── data/  # fetch_data.py
│   ├── processing/ # preprocessing.py
│   └── models/ # all train_model.py & predict,py
│
├── main.py
└── requirements.txt
```

---

# 📄 File Explanation

- **main.py** → runs full pipeline  
- **fetch_data.py** → collects data  
- **preprocess.py** → cleans data  
- **train models** → trains ML models  
- **predict.py** → generates predictions  
- **app.py** → dashboard  

---

# ⚙️ Installation

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

### Run Pipeline
```bash
python main.py
```

### Run Dashboard
```bash
streamlit run app/app.py
```

---

# 📊 Dashboard Features

- 📈 Live charts  
- 📉 Price indicators  
- 💰 Multi-asset tracking  
- 💱 Currency tracking  
- 📊 Interactive UI  
- 📁 Report export  

---

# 🔮 Prediction System

- Next-day prediction  
- Multi-asset prediction  
- Uses lag + moving average  

---

# 🚀 Future Improvements

- LSTM / Deep Learning  
- Cloud deployment  
- Real-time streaming  
- Portfolio optimization  

---

# 👨‍💻 Author

Daksh Vasani  
Data Science Student 

---

# ⭐ Support

If you like this project, give it a ⭐ on GitHub!

<p align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,100:2c5364&height=170&section=footer&text=Thanks%20for%20Visiting!&fontSize=28&fontColor=ffffff&animation=twinkling&fontAlignY=65"/>
</p>
