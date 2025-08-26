#%%
# --- Step 1: Setup, Load and Clean Data ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset and clean column names
df = pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip()
print("Dataset loaded successfully.")

# Impute missing values with the mean of their respective columns
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
print("Data cleaning complete.")


#%%
# ### --- REVISION 1: EXPLORE TEMPORAL TRENDS --- ###

# Set plotting style
sns.set_style('whitegrid')
plt.figure(figsize=(12, 6))

# Group by Year and Status to see the trend over time
temporal_trend = df.groupby(['Year', 'Status'])['Life expectancy'].mean().unstack()

# Plot the trend
temporal_trend.plot(kind='line', marker='o', figsize=(12, 6))
plt.title('Life Expectancy Trend: Developed vs. Developing Countries (2000-2015)', fontsize=16)
plt.ylabel('Average Life Expectancy (Age)')
plt.xlabel('Year')
plt.grid(True)
plt.legend(title='Status')
plt.show()


#%%
# --- Step 2: Data Preprocessing for Modeling ---

# Handle categorical 'Status' column and drop non-predictive 'Country' column
df_model = pd.get_dummies(df, columns=['Status'], drop_first=True)
df_model = df_model.drop('Country', axis=1)

# Define our features (X) and target (y)
X = df_model.drop('Life expectancy', axis=1)
y = df_model['Life expectancy']

print("\nData preprocessed for modeling.")


#%%
# ### --- REVISION 2: MODEL BENCHMARKING WITH CROSS-VALIDATION --- ###
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Define the models to benchmark
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

# Set up K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary to store the results
results = {}

print("\n--- Benchmarking Models with 5-Fold Cross-Validation ---")
# Loop through the models and perform cross-validation
for name, model in models.items():
    # We use R-squared as the scoring metric
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    results[name] = cv_scores.mean()
    print(f"{name}: Average R2 Score = {cv_scores.mean():.4f}")

# Find the best model based on average R2 score
best_model_name = max(results, key=results.get)
print(f"\nBest performing model: {best_model_name}")


#%%
# ### --- REVISION 3: FEATURE IMPORTANCE ANALYSIS --- ###
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Split the data to train the final best model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the best model from our benchmark
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# --- Final Evaluation on Test Set ---
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n--- Final Evaluation of {best_model_name} ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} years")
print(f"R-squared (R2) Score: {r2:.2f}")


# --- Get and Plot Feature Importances ---
# This works for tree-based models like Random Forest, Gradient Boosting, and XGBoost
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)

    # Plot the feature importances
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top 10 Most Important Features for {best_model_name}', fontsize=16)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()
# %%
