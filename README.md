# 🛒 Superstore Customer Intelligence Dashboard

> **An end-to-end Machine Learning & Analytics web application** — Customer Segmentation (RFM + K-Means) and Sales Forecasting (Holt-Winters), built with Python & Streamlit and deployed on Streamlit Community Cloud.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-9B59B6?logo=plotly)

---

## 📌 Project Overview

This project demonstrates a **complete DA workflow** on the classic **Superstore Sales Dataset (2014–2017)**:

| Module | Technique | Output |
|---|---|---|
| **Customer Segmentation** | RFM Feature Engineering + K-Means Clustering | Segment labels, 3D scatter, radar profile |
| **Sales Forecasting** | Holt-Winters Exponential Smoothing | 6–12 month revenue & profit forecast |

The dashboard is built with a **Modern Dark Theme** for a premium, portfolio-ready presentation.

---

## 🎯 Features

- **RFM Segmentation**: Automatically classifies customers into 3–7 segments based on purchase behavior.
- **Interactive K-Means**: Slider-driven cluster selection backed by real-time Elbow Curve and Silhouette Score.
- **3D Scatter Plot**: Plotly-powered 3D RFM visualization with hover details per customer.
- **Radar Profiles**: Per-segment normalized RFM radar chart for business storytelling.
- **Holt-Winters Forecasting**: Seasonal time-series model predicting up to 12 months ahead with confidence bands.
- **Searchable Customer Table**: Filter any individual customer's RFM profile and ML segment.

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit Dashboard entry-point
├── rfm_clustering.py         # RFM feature engineering + K-Means + Forecasting logic
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml           # Dark theme UI config
├── data/
│   ├── raw/
│   │   └── superstore_raw.csv
│   └── processed/
│       └── monthly_aggregation.csv
└── notebooks/                # Exploratory analysis (EDA) notebooks
```

---

## 🚀 Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/superstore-customer-intelligence.git
cd superstore-customer-intelligence

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501** 🎉

---

## 🧠 Machine Learning Methodology

### 1. RFM Feature Engineering
For each unique `Customer ID`, we compute:
- **Recency (R)** — Days since last purchase (lower = more engaged)
- **Frequency (F)** — Number of unique orders
- **Monetary (M)** — Total revenue generated (USD)

### 2. K-Means Clustering
- Features are standardized using `StandardScaler` before clustering.
- Optimal K is selected using the **Elbow Method (WCSS)** + **Silhouette Score**.
- Clusters are ranked by average Monetary value for consistent labeling.

### 3. Holt-Winters Forecasting
- Applies **additive trend + additive seasonality** (seasonal period = 12 months).
- Model parameters are optimized automatically using MLE.
- Confidence bands (±10%) are displayed alongside point forecasts.

---

## 📊 Dataset

| Field | Description |
|---|---|
| Source | [Kaggle — Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) |
| Period | January 2014 — December 2017 |
| Records | ~9,994 order-level rows |
| Columns | Order Date, Customer ID, Segment, Region, Sales, Profit, Discount, … |

---

## 🌐 Deployment

This app can be deployed for **free** on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Set **Main file path** to `app.py`.
4. Click **Deploy** — done! 🎉

---

## 🛠️ Tech Stack

`Python 3.11` · `Streamlit` · `scikit-learn` · `statsmodels` · `Plotly` · `Pandas` · `NumPy`

---

*Built as a DA Portfolio Project — showcasing end-to-end ML, business analytics, and data visualization skills.*
