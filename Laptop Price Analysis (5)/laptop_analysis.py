#%%
import pandas as pd
import numpy as np

# Load the dataset
try:
    df = pd.read_csv('laptop_price.csv', encoding='latin-1')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'laptop_price.csv' not found. Please make sure the file is in the correct project folder.")

# Display the first 5 rows of the dataframe
print("\nFirst 5 rows of the data:")
print(df.head())

# Get the number of rows and columns
print(f"\nDataset shape: {df.shape}")

# Get a summary of the dataset (columns, data types, non-null counts)
print("\nDataset Information:")
df.info()
#%%
import pandas as pd
import numpy as np

# --- Data Cleaning and Preprocessing (Corrected) ---

# Note: The line dropping 'Unnamed: 0' has been removed as it's not needed.

# Clean 'Ram' column by removing 'GB' and converting to integer
df['Ram'] = df['Ram'].str.replace('GB','').astype('int32')

# Clean 'Weight' column by removing 'kg' and converting to float
df['Weight'] = df['Weight'].str.replace('kg','').astype('float32')

# --- Feature Engineering from 'ScreenResolution' ---

# Check for Touchscreen
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)

# Check for IPS panel
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

# Extract screen resolution to calculate pixels per inch (PPI)
def find_x_res(s):
    return s.split()[-1].split('x')[0]

def find_y_res(s):
    return s.split()[-1].split('x')[1]

df['X_res'] = df['ScreenResolution'].apply(find_x_res)
df['Y_res'] = df['ScreenResolution'].apply(find_y_res)

# Convert resolution columns to numeric
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

# --- Feature Engineering from 'Cpu' and 'Gpu' ---

# Simplify CPU names
df['Cpu_brand'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

def categorize_processor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    elif text.split()[0] == 'Intel':
        return 'Other Intel Processor'
    else:
        return 'AMD Processor'

df['Cpu_brand'] = df['Cpu_brand'].apply(categorize_processor)


# Simplify GPU names
df['Gpu_brand'] = df['Gpu'].apply(lambda x: x.split()[0])


# --- Final Cleanup ---

# Drop the original complex columns that we have replaced
df.drop(columns=['Cpu','ScreenResolution','Gpu'], inplace=True)

print("Data cleaning and feature engineering complete!")
print("\nCleaned DataFrame Information:")
df.info()

print("\nFirst 5 rows of the cleaned data:")
df.head()
#%%
import matplotlib.pyplot as plt
import seaborn as sns

# --- Exploratory Data Analysis (EDA) ---

# Set the plotting style
sns.set_style('whitegrid')

# Create a figure for our plots
plt.figure(figsize=(18, 12))
plt.suptitle('Exploratory Data Analysis of Laptop Price Factors', fontsize=20)


# --- Plot 1: Distribution of Laptop Prices ---
plt.subplot(2, 2, 1)
sns.histplot(df['Price_euros'], kde=True, color='purple')
plt.title('Distribution of Laptop Prices', fontsize=14)
plt.xlabel('Price (Euros)')
plt.ylabel('Frequency')


# --- Plot 2: Average Price by Company ---
plt.subplot(2, 2, 2)
# Order the bars by price
brand_price = df.groupby('Company')['Price_euros'].mean().sort_values(ascending=False)
sns.barplot(x=brand_price.index, y=brand_price.values, palette='viridis')
plt.xticks(rotation=90)
plt.title('Average Price by Brand', fontsize=14)
plt.xlabel('Brand')
plt.ylabel('Average Price (Euros)')


# --- Plot 3: RAM vs. Price ---
plt.subplot(2, 2, 3)
sns.barplot(x=df['Ram'], y=df['Price_euros'], palette='magma')
plt.title('RAM vs. Price', fontsize=14)
plt.xlabel('RAM (in GB)')
plt.ylabel('Price (Euros)')


# --- Plot 4: Correlation Heatmap ---
plt.subplot(2, 2, 4)
# Create a correlation matrix of only the numeric columns
numeric_cols = df.select_dtypes(include=np.number)
correlation_matrix = numeric_cols.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features', fontsize=14)


# Adjust layout and show the plots
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
#%%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# --- Final Preprocessing ---

# Select all columns except the target variable for our features (X)
X = df.drop(columns=['Price_euros'])
# Select the target variable (y)
y = df['Price_euros']

# One-hot encode all categorical columns in X
# This will automatically find and convert columns with text data
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("Data successfully preprocessed and split for modeling.")

# --- Model Building and Evaluation ---

# Initialize and train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("\nTraining the Random Forest Regressor model...")
rf_model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"R-squared (R2) Score: {r2:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} Euros")

# --- Visualize Predictions vs Actual Values ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
# Plot a line for perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Prices', fontsize=16)
plt.xlabel('Actual Price (Euros)')
plt.ylabel('Predicted Price (Euros)')
plt.show()

# %%
