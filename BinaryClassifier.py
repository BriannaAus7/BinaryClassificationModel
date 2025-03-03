# Import necessary libraries and packages.
import numpy as np
import os
import joblib
import pandas as pd
import shap
import optuna
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


# 1. Function to read and flatten data.
def read_and_flatten(file_path):
    df = pd.read_csv(file_path, header=None)
    brain_regions = df.iloc[0, 1:].tolist()
    affinity_matrix = df.iloc[1:, 1:].values.astype(float)
    return affinity_matrix.flatten(), brain_regions


# Define directories for data
cue_hc_dir = "/Users/briannaaustin/Desktop/lsngc/EC_Brianna(3)/CueData-2/HC_Cue"
cue_patients_dir = "/Users/briannaaustin/Desktop/lsngc/EC_Brianna(3)/CueData-2/Patients_Cue"
mid_hc_dir = "/Users/briannaaustin/Desktop/lsngc/EC_Brianna(3)/MIDData/HC_MID"
mid_p_dir = "/Users/briannaaustin/Desktop/lsngc/EC_Brianna(3)/MIDData/Patients_MID"

# Initialize lists to hold the data and labels
X = []
y = []
brain_regions = None


# Process files in a directory
def process_directory(directory, label):
    global brain_regions
    for filename in os.listdir(directory):
        if filename.endswith("_Aff.csv"):
            filepath = os.path.join(directory, filename)
            flattened_matrix, regions = read_and_flatten(filepath)
            X.append(flattened_matrix)
            y.append(label)
            if brain_regions is None:
                brain_regions = regions


# Process all directories
process_directory(cue_hc_dir, 0)
process_directory(cue_patients_dir, 1)
process_directory(mid_hc_dir, 0)
process_directory(mid_p_dir, 1)

# Convert to numpy arrays
X = np.array(X, dtype=float)
y = np.array(y, dtype=int)

# Handle NaN values
col_mean = np.nanmean(X, axis=0)
inds = np.where(np.isnan(X))
X[inds] = np.take(col_mean, inds[1])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
best_k = 50
selector = SelectKBest(f_classif, k=best_k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = selector.get_support(indices=True)
feature_names = [f"{brain_regions[i]}_{brain_regions[j]}" for i in range(len(brain_regions)) for j in
                 range(len(brain_regions))]
selected_feature_names = [feature_names[i] for i in selected_features]

# Oversample with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)


# Optuna optimization for each model
def objective(trial):
    # Select scaling technique
    scaler_name = trial.suggest_categorical('scaler', [
        'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'QuantileTransformer'
    ])

    # Select model
    model_name = trial.suggest_categorical('model', [
        'Random Forest', 'XGBoost', 'LightGBM', 'Decision Tree',
        'Gradient Boosting', 'AdaBoost', 'Logistic Regression'
    ])

    # Select appropriate scaler
    if scaler_name == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_name == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_name == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')

    # Model-specific hyperparameters
    if model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            random_state=42
        )

    elif model_name == 'XGBoost':
        model = XGBClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            random_state=42,
            n_jobs=-1
        )

    elif model_name == 'LightGBM':
        model = LGBMClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            num_leaves=trial.suggest_int('num_leaves', 20, 200),
            min_child_samples=trial.suggest_int('min_child_samples', 1, 50),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )

    elif model_name == 'Decision Tree':
        model = DecisionTreeClassifier(
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            random_state=42
        )

    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
            random_state=42
        )

    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier(
            n_estimators=trial.suggest_int('n_estimators', 50, 500),
            learning_rate=trial.suggest_float('learning_rate', 0.01, 1.0),
            random_state=42
        )

    elif model_name == 'Logistic Regression':
        model = LogisticRegression(
            C=trial.suggest_float('C', 0.01, 10.0),
            max_iter=1000,
            random_state=42
        )

    # Create pipeline with scaling and model
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])

    # Train and evaluate the model using cross-validation
    cv_scores = cross_val_score(pipeline, X_train_selected, y_train, cv=5, scoring='accuracy')
    return np.mean(cv_scores)


# Optimize with more trials and TPE sampler.
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(n_startup_trials=20)
)
study.optimize(objective, n_trials=200)

# Print best trial information
print("Best trial:")
trial = study.best_trial
print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Create the best model based on the optimal configuration
best_model_name = trial.params['model']
best_scaler_name = trial.params['scaler']

# Select appropriate scaler
if best_scaler_name == 'StandardScaler':
    best_scaler = StandardScaler()
elif best_scaler_name == 'RobustScaler':
    best_scaler = RobustScaler()
elif best_scaler_name == 'MinMaxScaler':
    best_scaler = MinMaxScaler()
else:
    best_scaler = QuantileTransformer(n_quantiles=100, output_distribution='normal')

# Remove scaler and model params from trial params
best_params = {key: value for key, value in trial.params.items() if key not in ['model', 'scaler']}

# Create the best model
if best_model_name == 'Random Forest':
    best_model = RandomForestClassifier(random_state=42, **best_params)
elif best_model_name == 'XGBoost':
    best_model = XGBClassifier(random_state=42, n_jobs=-1, **best_params)
elif best_model_name == 'LightGBM':
    best_model = LGBMClassifier(random_state=42, **best_params)
elif best_model_name == 'Decision Tree':
    best_model = DecisionTreeClassifier(random_state=42, **best_params)
elif best_model_name == 'Gradient Boosting':
    best_model = GradientBoostingClassifier(random_state=42, **best_params)
elif best_model_name == 'AdaBoost':
    best_model = AdaBoostClassifier(random_state=42, **best_params)
elif best_model_name == 'Logistic Regression':
    best_model = LogisticRegression(random_state=42, max_iter=1000, **best_params)

# Scale the data with the best scaler
X_train_scaled = best_scaler.fit_transform(X_train_selected)
X_test_scaled = best_scaler.transform(X_test_selected)

# Train the best model
best_model.fit(X_train_scaled, y_train)

# Evaluate on the test set
y_pred = best_model.predict(X_test_scaled)
y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# Classification Report
print(f"\nBest Model: {best_model_name}")
print(f"Best Scaler: {best_scaler_name}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve: {best_model_name}')
plt.legend(loc="lower right")
plt.show()

# SHAP Analysis
explainer = shap.Explainer(best_model, X_test_scaled)
shap_values = explainer(X_test_scaled)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=selected_feature_names)

# Save the best model
model_save_path = f'best_model_{best_k}_features_{best_model_name}_{best_scaler_name}.pkl'
joblib.dump(best_model, model_save_path)
print(f"\nModel saved successfully at {model_save_path}")



# Top 10 most important features
def analyze_shap_values(shap_values, feature_names):
    # Calculate the mean absolute SHAP values for each feature
    shap_importance = np.abs(shap_values.values).mean(axis=0)

    # Sort features by their importance
    feature_importance = sorted(zip(feature_names, shap_importance),
                                key=lambda x: x[1],
                                reverse=True)

    # Print top 10 most important features
    print("Top 10 Most Important Features:")
    for name, importance in feature_importance[:10]:
        print(f"{name}: {importance:.4f}")

    # Visualize top 10 feature importances
    plt.figure(figsize=(10, 6))
    top_features = feature_importance[:10]
    plt.barh([name for name, _ in top_features],
             [importance for _, importance in top_features])
    plt.title('Top 10 Features by Mean Absolute SHAP Value')
    plt.xlabel('Mean Absolute SHAP Value')
    plt.ylabel('Feature Name')
    plt.tight_layout()
    plt.show()

    return feature_importance


# Perform the analysis
feature_importance = analyze_shap_values(shap_values, selected_feature_names)

# Additional SHAP plot for detailed visualization
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_scaled, feature_names=selected_feature_names)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.show() 


#Notes for code improvement 
## 1. Speed up Optuna with parallelization (n_jobs=-1)
## 2. Optimise memory usage using scripy.sparse
## 3. Save Optuna trials so tuning can be resumed (sqlite)
## 4. Try using RFE instead of SelectKBest
## 5. Improve SHAP analysis to include bar charts.
