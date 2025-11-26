# Data Analyst Project Portfolio

### Introduction
Welcome to my portfolio! This repository showcases a collection of six comprehensive data analyst and machine learning projects completed in August 2025. Each project demonstrates an end-to-end data science workflow, from data cleaning and feature engineering to model building, iterative refinement, and final evaluation.

---

### Technical Skills

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-006699?style=for-the-badge&logo=xgboost&logoColor=white)
![Prophet](https://img.shields.io/badge/Prophet-0072B2?style=for-the-badge&logo=facebook&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-88d4dd?style=for-the-badge&logo=seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=matplotlib&logoColor=white)

---

### Projects

#### Uber Trip Analysis
* **Goal**: 📈 Forecast hourly Uber trip demand in New York City to improve operational efficiency.
* **Method**: Developed an iterative forecasting solution, incorporating external weather data and advanced time-based features.
* **Result**: The final Ensemble model **outperformed a naïve baseline by over 70%**, achieving a robust MAPE of 10.03%.
* **Skills**: Time-Series Forecasting, External Data Integration, Ensemble Modeling, Baseline Comparison.

#### TCS Stock Price Prediction
* **Goal**: 💹 Build a deep learning model to forecast daily returns for TCS stock, a more meaningful target than price.
* **Method**: Constructed and trained an LSTM neural network with an advanced training regimen (`EarlyStopping`).
* **Result**: The LSTM model's performance **(RMSE: 0.0127) was rigorously benchmarked and proven to be on par with a classic ARIMA model**.
* **Skills**: Deep Learning (LSTM), Financial Forecasting (Returns), Model Benchmarking (ARIMA, Prophet), Keras Callbacks.

#### Personalized Healthcare Recommendations
* **Goal**: ❤️ Develop a production-ready model to predict a patient's risk of heart disease.
* **Method**: Built a complete Scikit-learn Pipeline with an XGBoost classifier, validated with Stratified K-Fold Cross-Validation.
* **Result**: Achieved a **cross-validated accuracy of 79.22%** and a strong **ROC-AUC score of 0.86**. Model predictions were explained using **SHAP**.
* **Skills**: Classification (XGBoost), Scikit-learn Pipelines, Cross-Validation, Model Interpretability (SHAP).

#### Life Expectancy Analysis
* **Goal**: 🌍 Predict a country's life expectancy and identify the key socio-economic and health drivers.
* **Method**: Benchmarked four different regression models using K-Fold Cross-Validation.
* **Result**: The champion Random Forest model achieved a **cross-validated R-squared score of 0.96**. Feature importance analysis identified key drivers like income and mortality rates.
* **Skills**: Regression, Model Benchmarking, Cross-Validation, Feature Importance.

#### Laptop Price Analysis
* **Goal**: 💻 Create a model to predict laptop prices and explain what features drive the cost.
* **Method**: Engineered new features (Pixels Per Inch), analyzed outliers, and benchmarked four regression models.
* **Result**: The champion model was selected after rigorous benchmarking, and feature importance analysis identified RAM and PPI as key price drivers.
* **Skills**: Regression, Feature Engineering, Model Benchmarking, Explainability.

#### World Population Forecast
* **Goal**: 🌐 Analyze annual historical data to forecast the total world population up to the year 2050.
* **Method**: Benchmarked Prophet, ARIMA, and LSTM models on granular, annual data from 1960-2022.
* **Result**: The **Prophet model was selected as the champion** after proving its superior accuracy in the benchmark.
* **Skills**: Time-Series Forecasting, Model Benchmarking (Prophet, ARIMA, LSTM), Data Granularity.
