# Global Air Pollution Forecasting Model

<br>
<br>
<p align="center">
🌐 Try Live Demo <a href="https://global-air-pollution-forecasting-model-cwr4glzywgq2jhhkhdrrhq.streamlit.app/" target="_blank"></a>
</p>

---

## 🔹 Project Overview

The **Global Air Pollution Forecasting Model** predicts PM2.5 AQI values using **machine learning**. It helps users monitor air quality in real time across **174+ countries** with an interactive, easy-to-use web interface.

**Objectives:**
- Provide accurate PM2.5 AQI predictions.
- Visualize AQI trends using intuitive charts.
- Compare model performance between Random Forest and Linear Regression.

---

## ✨ Key Features

- **AI Predictions:** Random Forest model with 71.5% accuracy.  
- **Global Coverage:** Supports 174 countries.  
- **Visual Analytics:** Gauge charts and color-coded AQI categories.  
- **Model Comparison:** Random Forest vs Linear Regression.  
- **Instant Results:** Predictions in milliseconds.  
- **Interactive Web App:** Built with Streamlit.

---


## 📁 Project Structure

```
📦 Global-Air-Pollution-Forecasting-Model/
├── streamlit_app.py              # Interactive web application
├── requirements.txt              # Python dependencies
├── 📂 models/                       # Trained ML models
│   ├── random_forest_air_pollution_model.joblib
│   ├── linear_regression_air_pollution_model.joblib
│   └── feature_names.joblib
├── 📂 notebooks/                    # Jupyter notebooks
│   └── global_air_pollution_model.ipynb
├── 📂 data/                         # Dataset
│   └── global air pollution dataset.csv
├── 📂 src/                          # Source code
│   ├── global_air_pollution_model.py
│   └── model_inference.py
└── README.md                     # This file
```

## 🤖 Model Performance

### 🌟 Random Forest (Recommended)
- **R² Score:** `0.7152` (71.5% variance explained)
- **Mean Absolute Error:** `16.06`
- **RMSE:** `30.17`
- **Features:** 177 (3 numerical + 174 countries)
- **Trees:** 100

### 📉 Linear Regression (Comparison)
- **R² Score:** `Poor` (Negative value)
- **MAE:** `211,544,538.94`
- **Status:** ❌ Not recommended

## 📊 Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Records** | 23,035 |
| **Countries** | 174 |  
| **Features** | 177 |
| **Target** | PM2.5 AQI Value |
| **Missing Data** | Cleaned (1.8% removed) |

### Input Features:
- **CO AQI Value** - Carbon Monoxide levels
- **Ozone AQI Value** - Ground-level ozone
- **NO2 AQI Value** - Nitrogen dioxide levels
- **Country** - Geographic location (174 countries)

## Setup Instructions

### 1. Web Application (Easiest)
Visit the [**Live Demo**](https://global-air-pollution-forecasting-model-cwr4glzywgq2jhhkhdrrhq.streamlit.app/) for instant predictions!

### 2. Local Installation
```bash
# Clone repository
git clone 

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app

```

### 3. Python API Usage
```python
import joblib
import pandas as pd

# Load models
rf_model = joblib.load('models/random_forest_air_pollution_model.joblib')
feature_names = joblib.load('models/feature_names.joblib')

# Make prediction
def predict_air_quality(co_aqi=1, ozone_aqi=35, no2_aqi=3, country='USA'):
    input_data = pd.DataFrame(columns=feature_names).fillna(0)
    input_data.loc[0, 'CO AQI Value'] = co_aqi
    input_data.loc[0, 'Ozone AQI Value'] = ozone_aqi  
    input_data.loc[0, 'NO2 AQI Value'] = no2_aqi
    
    if f'Country_{country}' in feature_names:
        input_data.loc[0, f'Country_{country}'] = 1
    
    return rf_model.predict(input_data)[0]

# Example
pm25_prediction = predict_air_quality(co_aqi=2, ozone_aqi=45, no2_aqi=5, country='China')
print(f"Predicted PM2.5 AQI: {pm25_prediction:.1f}")
```

## 🌬️ What It Does

This web application forecasts the PM2.5 Air Quality Index (AQI) based on environmental indicators and country-specific data. By leveraging machine learning, it helps users understand air pollution trends and potential health impacts.

---

## 🧪 AQI Reference Table

| AQI Value | Description                  | Color | Health Advisory                  |
|-----------|-------------------------------|-------|---------------------------------|
| 0-50      | Excellent                     | 🟢    | Air quality is considered safe  |
| 51-100    | Fair                          | 🟡    | Minor health concerns possible  |
| 101-150   | Sensitive Population Alert    | 🟠    | Vulnerable individuals affected |
| 151-200   | Unhealthy                     | 🔴    | Everyone may experience effects |
| 201-300   | Very Unhealthy                | 🟣    | Health alert; reduce outdoor activity |
| 301+      | Hazardous                     | 🔴    | Emergency conditions; stay indoors |

---

## ⚙️ How It Works

1. **Clean & Prepare Data** – Handle missing values and remove irrelevant columns  
2. **Feature Transformation** – One-hot encode country information  
3. **Model Training** – Random Forest regression with 100 decision trees  
4. **Evaluation** – Validate performance using 80/20 train-test split  

---

## 📊 Model Performance

| Model             | R² Score | MAE    | RMSE   | Status             |
|------------------|----------|--------|--------|------------------|
| Random Forest     | 0.715    | 16.06  | 30.17  | ✅ Reliable       |
| Linear Regression | Poor     | 211M+  | 14B+   | ❌ Not Recommended |

---

## 🛠️ Tech Stack

- **Backend & ML:** Python, pandas, numpy, scikit-learn  
- **Frontend & Visualization:** Streamlit, Plotly  
- **Deployment:** Streamlit Cloud  
- **Version Control:** Git, GitHub  

---

## ✨ Key Features

- **Global Coverage** – Predict PM2.5 for 174 countries  
- **Interactive Charts** – Visualize AQI trends dynamically  
- **Export Predictions** – Download results in CSV or JSON  
- **User-Friendly Interface** – Clean layout for all users  

---

## 📜 License & Credits

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Acknowledgments:**  
- Contributors of global air pollution datasets  
- Streamlit and scikit-learn communities  
- Plotly for advanced visualizations  

---
