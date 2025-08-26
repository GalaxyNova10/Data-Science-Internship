#%%
# --- Step 1: Setup and Data Retrieval ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Download data
tcs_ticker = 'TCS.NS'
tcs_data = yf.download(tcs_ticker, start='2010-01-01')
print("Dataset loaded successfully.")


# ### --- REVISION 1: SWITCH TARGET TO DAILY RETURNS --- ###
returns_data = tcs_data[['Close']].pct_change().dropna()
print("\nSwitched target to daily returns.")


# Split data into training and testing sets (80/20 split)
training_data_len = int(np.ceil(len(returns_data) * .8))
train_data = returns_data[:training_data_len]
test_data = returns_data[training_data_len:]


#%%
# ### --- REVISION 2: ADD COMPLETE BASELINES (NAÏVE, PROPHET, ARIMA) --- ###

print("\n--- Generating Baseline Forecasts ---")
y_true = test_data.values

# --- Naïve Baseline (predict today's return is same as yesterday's) ---
y_pred_naive = train_data.values[-1] # Predict last known return for the whole test period
naive_rmse = np.sqrt(mean_squared_error(y_true, [y_pred_naive]*len(y_true)))
print(f"Naïve Baseline RMSE: {naive_rmse:.6f}")

# --- Prophet Baseline ---
prophet_train_df = train_data.reset_index()
prophet_train_df.columns = ['ds', 'y']
prophet_model = Prophet(daily_seasonality=True)
prophet_model.fit(prophet_train_df)
future = prophet_model.make_future_dataframe(periods=len(test_data), freq='B')
forecast = prophet_model.predict(future)
y_pred_prophet = forecast[-len(test_data):]['yhat'].values
prophet_rmse = np.sqrt(mean_squared_error(y_true, y_pred_prophet))
print(f"Prophet Baseline RMSE: {prophet_rmse:.6f}")

# --- ARIMA Baseline ---
# Using auto_arima to find the best ARIMA model
auto_arima_model = pm.auto_arima(train_data, seasonal=False, stepwise=True,
                                 suppress_warnings=True, error_action='ignore', max_p=5, max_q=5)
print(f"Best ARIMA Order: {auto_arima_model.order}")
y_pred_arima = auto_arima_model.predict(n_periods=len(test_data))
arima_rmse = np.sqrt(mean_squared_error(y_true, y_pred_arima))
print(f"ARIMA Baseline RMSE: {arima_rmse:.6f}")


#%%
# --- Step 3: LSTM Data Preprocessing for Returns ---
dataset = returns_data.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

train_data_scaled = scaled_data[0:int(training_data_len), :]
x_train, y_train = [], []
for i in range(60, len(train_data_scaled)):
    x_train.append(train_data_scaled[i-60:i, 0])
    y_train.append(train_data_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print("\nLSTM training data preprocessed for returns.")


#%%
# ### --- REVISION 4: TUNED LSTM ARCHITECTURE AND ADVANCED TRAINING --- ###

# --- Build a slightly larger LSTM Model ---
model = Sequential()
# Increased units from 50 to 100
model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# --- Define Callbacks for Smart Training ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# --- Train the Model ---
print("\nStarting advanced LSTM training on returns... (This will take several minutes)")
history = model.fit(x_train, y_train, batch_size=64, epochs=50,
                    validation_split=0.1, callbacks=[early_stopping, reduce_lr])
print("Advanced LSTM training complete.")


#%%
# ### --- REVISION 5: ENHANCED EVALUATION on Returns --- ###

# Prepare Test Data
test_data_scaled = scaled_data[training_data_len - 60:, :]
x_test = []
y_test_true_original = dataset[training_data_len:, :]
for i in range(60, len(test_data_scaled)):
    x_test.append(test_data_scaled[i-60:i, 0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make Predictions
predictions_scaled = model.predict(x_test)
predictions = scaler.inverse_transform(predictions_scaled)

# Calculate All Metrics
lstm_rmse = np.sqrt(mean_squared_error(y_test_true_original, predictions))
lstm_mae = mean_absolute_error(y_test_true_original, predictions)
lstm_mape = mean_absolute_percentage_error(y_test_true_original, predictions)

print("\n--- Final Model Evaluation on Daily Returns ---")
print(f"Naïve Baseline RMSE: {naive_rmse:.6f}")
print(f"Prophet Baseline RMSE: {prophet_rmse:.6f}")
print(f"ARIMA Baseline RMSE: {arima_rmse:.6f}")
print("---------------------------------------------")
print(f"Advanced LSTM RMSE: {lstm_rmse:.6f}")
print(f"Advanced LSTM MAE: {lstm_mae:.6f}")
print(f"Advanced LSTM MAPE: {lstm_mape:.2%}")

#%%
# --- Final Visualization (Actual vs. Predicted Returns) ---
import seaborn as sns # <-- This is the missing import line
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_true_original.flatten(), y=predictions.flatten(), alpha=0.5)
plt.title('Actual vs. Predicted Daily Returns', fontsize=16)
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.axline([0,0], slope=1, color='r', linestyle='--') # Add a line for perfect predictions
plt.grid(True)
plt.show()
# %%
