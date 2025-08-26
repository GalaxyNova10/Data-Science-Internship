#%%
# --- Step 1: Setup and Data Loading ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import holidays

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# --- Load Uber Data ---
data_path = 'data/'
files = [
    'uber-raw-data-apr14.csv','uber-raw-data-may14.csv','uber-raw-data-jun14.csv',
    'uber-raw-data-jul14.csv','uber-raw-data-aug14.csv','uber-raw-data-sep14.csv'
]
dataframes = [pd.read_csv(os.path.join(data_path, file)) for file in files]
uber_2014 = pd.concat(dataframes, ignore_index=True)
print("Uber dataset loaded successfully!")

# --- Load Weather Data ---
try:
    weather_df = pd.read_csv('weather_description.csv')
    print("Weather dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'weather_description.csv' not found. Please make sure the file is in the project folder.")

#%%
# --- Step 2: Data Preparation and Merging ---

# Prepare Uber Data
uber_2014['Date/Time'] = pd.to_datetime(uber_2014['Date/Time'], format='%m/%d/%Y %H:%M:%S')
uber_2014.set_index('Date/Time', inplace=True)
uber_hourly = uber_2014['Base'].resample('h').count().to_frame(name='Count')

# Prepare Weather Data
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
weather_df = weather_df.set_index('datetime')

# Correctly select the 'New York' column which contains its weather data
weather_nyc = weather_df[['New York']].copy()
weather_nyc.rename(columns={'New York': 'Weather'}, inplace=True)

# Resample to hourly frequency and get the most common weather description
weather_hourly = weather_nyc.resample('h').agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# Merge Uber and Weather data
merged_df = pd.merge(uber_hourly, weather_hourly, left_index=True, right_index=True, how='left')

# Forward-fill any missing weather data points
merged_df['Weather'].fillna(method='ffill', inplace=True)

print("Uber and Weather data successfully merged.")
print(merged_df.head())

#%%
# ### --- Step 3: Advanced Feature Engineering --- ###
us_holidays = holidays.US()

def add_advanced_features(df):
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['dayofyear'] = df.index.dayofyear
    df['is_holiday'] = df.index.map(lambda x: 1 if x in us_holidays else 0)

    # Cyclical features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24.0)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek']/6.0)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek']/6.0)

    # Rolling window features
    df['rolling_mean_3h'] = df['Count'].shift(1).rolling(window=3).mean()
    df['rolling_std_3h'] = df['Count'].shift(1).rolling(window=3).std()
    
    # Lag features
    for i in range(24, 49, 8):
        df[f'lag_{i}'] = df['Count'].shift(i)

    # One-hot encode the 'Weather' categorical feature
    df = pd.get_dummies(df, columns=['Weather'], drop_first=True)

    df = df.drop(['hour', 'dayofweek'], axis=1)
    return df

featured_df = add_advanced_features(merged_df.copy())
featured_df = featured_df.dropna()
print("Advanced features (including weather) created successfully.")


#%%
# --- Step 4: Data Splitting and Baseline ---
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

X = featured_df.drop('Count', axis=1)
y = featured_df['Count']

cutoff_date = '2014-09-15'
X_train = X.loc[X.index < cutoff_date]
y_train = y.loc[y.index < cutoff_date]
X_test = X.loc[X.index >= cutoff_date]
y_test = y.loc[y.index >= cutoff_date]

# Create a Seasonal Naïve Baseline
test_data_for_baseline = uber_hourly.loc[uber_hourly.index >= cutoff_date]
y_true_baseline = test_data_for_baseline['Count'].values
y_pred_baseline = test_data_for_baseline['Count'].shift(24).dropna().values
y_true_baseline = y_true_baseline[24:]
baseline_mape = mean_absolute_percentage_error(y_true_baseline, y_pred_baseline)
print(f"Seasonal Naïve Baseline MAPE: {baseline_mape:.2%}")


#%%
# --- Step 5: Model Training and Evaluation ---
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

tscv = TimeSeriesSplit(n_splits=5)

# XGBoost Model
print("\n--- Training XGBoost Model ---")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid={'n_estimators': [100, 300], 'max_depth': [3, 6]}, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
xgb_grid_search.fit(X_train, y_train)
xgb_predictions = xgb_grid_search.best_estimator_.predict(X_test)
xgb_mape = mean_absolute_percentage_error(y_test, xgb_predictions)
print(f"XGBoost MAPE: {xgb_mape:.2%}")

# Random Forest Model
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid={'n_estimators': [100, 200], 'max_depth': [10, 20]}, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
rf_grid_search.fit(X_train, y_train)
rf_predictions = rf_grid_search.best_estimator_.predict(X_test)
rf_mape = mean_absolute_percentage_error(y_test, rf_predictions)
print(f"Random Forest MAPE: {rf_mape:.2%}")

# Gradient Boosting Model
print("\n--- Training Gradient Boosting Model ---")
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_grid_search = GridSearchCV(estimator=gbr_model, param_grid={'n_estimators': [100, 300], 'max_depth': [3, 5]}, cv=tscv, scoring='neg_mean_absolute_percentage_error', n_jobs=-1, verbose=1)
gbr_grid_search.fit(X_train, y_train)
gbr_predictions = gbr_grid_search.best_estimator_.predict(X_test)
gbr_mape = mean_absolute_percentage_error(y_test, gbr_predictions)
print(f"GBTR MAPE: {gbr_mape:.2%}")


#%%
# --- Step 6: Ensemble and Final Visualization ---
mape_scores = np.array([xgb_mape, rf_mape, gbr_mape])
weights = 1 / mape_scores
weights /= np.sum(weights)

ensemble_predictions = (weights[0] * xgb_predictions + weights[1] * rf_predictions + weights[2] * gbr_predictions)
ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_predictions)

print("\n--- Final Model Comparison ---")
print(f"Ensemble MAPE (with Weather Data): {ensemble_mape:.2%}")
print(f"Seasonal Naïve Baseline MAPE: {baseline_mape:.2%}")

# Final Visualization
plt.figure(figsize=(18, 8))
plt.plot(y_test.index, y_test.values, label='Actual Trips (Test Set)', color='black', linewidth=2)
plt.plot(y_test.index, ensemble_predictions, label=f'Ensemble Predictions (MAPE: {ensemble_mape:.2%})', color='green', linestyle='-')
plt.title('Uber Trip Forecasting with Weather Data', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Number of Trips')
plt.legend()
plt.show()

# %%
