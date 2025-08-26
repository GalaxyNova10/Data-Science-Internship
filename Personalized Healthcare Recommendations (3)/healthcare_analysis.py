#%%
# --- Step 1 & 2: Load and Clean Data ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart_disease_ci.csv')

# Handle missing values by dropping rows
df.dropna(inplace=True)

# Simplify the target variable to be binary (0 = No Disease, 1 = Disease)
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

# Drop the original 'num' column and any other non-feature columns
df_model = df.drop(['num', 'id', 'dataset'], axis=1)

print("Dataset loaded and cleaned successfully.")

#%%
# --- Step 3: Define Features (X) and Target (y) ---
X = df_model.drop('target', axis=1)
y = df_model['target']

# Identify categorical features for preprocessing
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print("Features and target defined.")

#%%
# ### --- REVISION 1: CREATE A DEPLOYMENT-READY PIPELINE --- ###
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Create a preprocessor object using ColumnTransformer
# This will apply one-hot encoding to categorical features and leave numerical features as they are
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Keep other columns (the numerical ones)
)

# Create the full pipeline by combining the preprocessor and the XGBoost model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'))
])

print("Preprocessing and modeling pipeline created successfully.")

#%%
# ### --- REVISION 2: IMPLEMENT ROBUST VALIDATION STRATEGY --- ###
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Use Stratified K-Fold cross-validation to get a reliable estimate of model accuracy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model_pipeline, X, y, cv=cv, scoring='accuracy')

print("\n--- Stratified K-Fold Cross-Validation ---")
print(f"Cross-Validation Accuracy Scores: {np.round(cv_scores, 2)}")
print(f"Average CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Standard Deviation of CV Accuracy: {cv_scores.std() * 100:.2f}%")

#%%
# ### --- REVISION 3: ADD ADVANCED EVALUATION METRICS --- ###
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix

# We still need a single split to generate plots and reports
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the pipeline on the full training set
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)
y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1] # Probabilities for the positive class

# --- Calculate Metrics ---
print("\n--- Comprehensive Model Evaluation on Test Set ---")
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Classification Report (includes Precision, Recall/Sensitivity)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Heart Disease', 'Heart Disease']))

# Specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)
print(f"Specificity: {specificity:.2f}")

# --- Generate Plots ---
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall/Sensitivity)')
plt.legend()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
plt.subplot(1, 2, 2)
plt.plot(recall, precision, label='XGBoost')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall (Sensitivity)')
plt.ylabel('Precision')
plt.legend()

plt.tight_layout()
plt.show()

#%%
# ### --- REVISION 4: IMPLEMENT MODEL INTERPRETABILITY (Corrected) --- ###
import shap

# SHAP requires the model itself, not the pipeline with the preprocessor
xgb_model = model_pipeline.named_steps['classifier']

# And we need to get the preprocessed data that was fed to the model
feature_names = model_pipeline.named_steps['preprocessor'].get_feature_names_out()

# Create the processed DataFrames WITHOUT the .toarray() call
X_train_processed = pd.DataFrame(model_pipeline.named_steps['preprocessor'].transform(X_train), columns=feature_names)
X_test_processed = pd.DataFrame(model_pipeline.named_steps['preprocessor'].transform(X_test), columns=feature_names)


# Create SHAP explainer and calculate SHAP values
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_processed)

# --- Generate SHAP Summary Plot ---
print("\n--- Model Interpretability with SHAP ---")
print("This plot shows the impact of each feature on the model's predictions.")
shap.summary_plot(shap_values, X_test_processed, plot_type="bar", show=False)
plt.title("Feature Importance based on SHAP Values")
plt.show()

shap.summary_plot(shap_values, X_test_processed)

# %%
