#%%
# --- Step 1: Setup, Load and Prepare Data ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error # <-- This is the missing line

# Load the new, correct annual dataset
df = pd.read_csv('global_population_trends.csv')
print("Annual population dataset loaded successfully.")

# --- Data Preparation ---
# Dynamically find the year columns and the identifier columns
population_cols = [col for col in df.columns if col.isdigit()]
id_vars = [col for col in df.columns if col not in population_cols]

# Melt the dataframe to turn year columns into a time series format
long_df = df.melt(id_vars=id_vars, value_vars=population_cols,
                  var_name='Year', value_name='Population')

# Sum all countries to get the total world population for each year
world_pop_ts = long_df.groupby('Year')['Population'].sum().to_frame().reset_index()

# Convert to datetime and sort
world_pop_ts['Year'] = pd.to_datetime(world_pop_ts['Year'], format='%Y')
world_pop_ts.sort_values('Year', inplace=True)
world_pop_ts.dropna(subset=['Population'], inplace=True) # Drop years with no population data

# Fulfills "Units Clarity" requirement
world_pop_ts['Population'] = world_pop_ts['Population'] / 1e9

print("\nAnnual World Population Data (in Billions):")
print(world_pop_ts.head())

# Split data into training and testing sets (test on the last 10 years)
train_data = world_pop_ts.iloc[:-10]
test_data = world_pop_ts.iloc[-10:]

#%%
# ### --- REVISION: MODEL BENCHMARKING (PROPHET, ARIMA, LSTM) --- ###

results = {} # Dictionary to store model performance

# --- Model 1: Prophet ---
from prophet import Prophet

print("\n--- Training Prophet Model ---")
prophet_df = train_data.rename(columns={'Year': 'ds', 'Population': 'y'})
prophet_model = Prophet(yearly_seasonality=True)
prophet_model.fit(prophet_df)
future = prophet_model.make_future_dataframe(periods=len(test_data), freq='AS') # 'AS' for annual start
forecast = prophet_model.predict(future)
y_pred_prophet = forecast[-len(test_data):]['yhat'].values

results['Prophet'] = {
    'RMSE': np.sqrt(mean_squared_error(test_data['Population'], y_pred_prophet)),
    'MAE': mean_absolute_error(test_data['Population'], y_pred_prophet)
}

# --- Model 2: ARIMA ---
import pmdarima as pm

print("\n--- Training ARIMA Model ---")
arima_train_data = train_data.set_index('Year')['Population']
auto_arima_model = pm.auto_arima(arima_train_data, seasonal=False, stepwise=True,
                                 suppress_warnings=True, error_action='ignore')
y_pred_arima = auto_arima_model.predict(n_periods=len(test_data))

results['ARIMA'] = {
    'RMSE': np.sqrt(mean_squared_error(test_data['Population'], y_pred_arima)),
    'MAE': mean_absolute_error(test_data['Population'], y_pred_arima)
}

# --- Model 3: LSTM ---
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping

print("\n--- Training LSTM Model ---")
# Scale data
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data.set_index('Year'))
scaled_test = scaler.transform(test_data.set_index('Year'))

# Create sequences
def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 5 # Use last 5 years to predict the next
X_train, y_train = create_sequences(scaled_train, n_steps)

# Build and train LSTM
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lstm_model.fit(X_train, y_train, epochs=100, validation_split=0.1, callbacks=[early_stopping], verbose=0)

# Make predictions with the LSTM
last_sequence = scaled_train[-n_steps:]
current_batch = last_sequence.reshape((1, n_steps, 1))
y_pred_lstm_scaled = []
for i in range(len(test_data)):
    current_pred = lstm_model.predict(current_batch, verbose=0)[0]
    y_pred_lstm_scaled.append(current_pred)
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

y_pred_lstm = scaler.inverse_transform(y_pred_lstm_scaled)

results['LSTM'] = {
    'RMSE': np.sqrt(mean_squared_error(test_data['Population'], y_pred_lstm)),
    'MAE': mean_absolute_error(test_data['Population'], y_pred_lstm)
}


#%%
# --- Step 3: Final Evaluation and Visualization ---
print("\n--- Model Benchmarking Results (Error in Billions of People) ---")
results_df = pd.DataFrame(results).T
print(results_df)

best_model_name = results_df['RMSE'].idxmin()
print(f"\nBest performing model based on RMSE: {best_model_name}")

# --- Final Forecast using the Full Dataset with Prophet ---
print("\nGenerating final forecast with champion model (Prophet) on all data...")
final_prophet_df = world_pop_ts.rename(columns={'Year': 'ds', 'Population': 'y'})
final_model = Prophet(yearly_seasonality=True)
final_model.fit(final_prophet_df)

# Create a dataframe for future dates up to 2050
future_final = final_model.make_future_dataframe(periods=28, freq='Y')
forecast_final = final_model.predict(future_final)

# Fulfills "Uncertainty Analysis" requirement
plt.figure(figsize=(14, 7))
fig = final_model.plot(forecast_final)
plt.title('World Population Forecast to 2050 (with Uncertainty Interval)', fontsize=16)
plt.xlabel('Year')
plt.ylabel('Population (in Billions)')
# Add the actual test data points to the final plot for comparison
ax = fig.gca()
ax.plot(test_data['Year'], test_data['Population'], 'r.', markersize=10, label='Actual Recent Data')
plt.legend()
plt.show()

# Plot components to understand the trend and seasonality
print("\nProphet's trend and seasonality analysis:")
fig2 = final_model.plot_components(forecast_final)
plt.show()
# %%
