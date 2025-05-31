import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # For saving the model and scaler
import os # For managing paths
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit # Import StratifiedKFold and StratifiedShuffleSplit
from sklearn.impute import SimpleImputer # For handling NaNs in features
from sklearn.metrics import confusion_matrix, recall_score # For evaluation metrics
from scipy.optimize import minimize_scalar # For robust threshold optimization

# --- Configuration ---
INPUT_MERGED_DF_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/dataset_5day.csv'
OUTPUT_FINAL_CSV_PATH = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/Kfold/results_5day_OCSVM.csv'
DATA_DIR = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/Kfold'
MODEL_DIR = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/models' # Directory to save the model and scaler
FIGURE_DIR = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Plots/Kfold'
# Hyperparameter ranges for tuning
NU_VALUES = [0.005, 0.01, 0.02, 0.05, 0.1]
GAMMA_VALUES = ['scale', 0.1, 0.01, 0.001]

# Define weights for the custom evaluation metric
WEIGHT_RECALL = 0.5
WEIGHT_FPR = 0.5

# K-Fold Cross-Validation settings
N_SPLITS = 5 # Number of folds for cross-validation during tuning

# --- Custom Weighted Score Function ---
# This function calculates the evaluation score based on recall and false positive rate.
# It now operates on period-level true/predicted labels but is geared towards supply-level logic.
def calculate_weighted_score(y_true_periods, y_pred_periods, df_info_for_supply_level, weight_recall, weight_false_positive_rate):
    """
    Calculates a weighted score based on supply-level recall and false positive rate.
    
    Args:
        y_true_periods (pd.Series or np.array): True labels for periods (1 for non-regular, 0 for regular).
        y_pred_periods (pd.Series or np.array): Predicted labels for periods (1 for anomaly, 0 for normal).
        df_info_for_supply_level (pd.DataFrame): DataFrame containing 'Supply_ID' and 'Is_Non_Regular' for grouping.
                                                   Must align with y_true_periods and y_pred_periods indices.
        weight_recall (float): Weight for recall of non-regular supplies.
        weight_false_positive_rate (float): Weight for false positive rate (FPR).

    Returns:
        float: The combined weighted score. Higher is better.
    """
    if len(y_true_periods) == 0:
        return 0.0 # No periods to evaluate

    # Create a temporary DataFrame for supply-level grouping and evaluation
    temp_df_eval = df_info_for_supply_level.copy()
    temp_df_eval['y_true_period'] = y_true_periods
    temp_df_eval['y_pred_period'] = y_pred_periods

    # Determine true non-regular supplies: if *any* period for that supply is non-regular
    true_non_regular_supplies_map = temp_df_eval.groupby('Supply_ID')['y_true_period'].max() # Max will be 1 if any non-regular period, 0 otherwise

    # Determine predicted anomalous supplies: if *any* period for that supply is predicted anomalous
    predicted_anomalous_supplies_map = temp_df_eval.groupby('Supply_ID')['y_pred_period'].max() # Max will be 1 if any predicted anomaly

    # Align the true and predicted supply labels
    all_supply_ids = true_non_regular_supplies_map.index.union(predicted_anomalous_supplies_map.index)
    
    y_true_supply = true_non_regular_supplies_map.reindex(all_supply_ids, fill_value=0)
    y_pred_supply = predicted_anomalous_supplies_map.reindex(all_supply_ids, fill_value=0)

    # Calculate confusion matrix components at the supply level
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true_supply, y_pred_supply, labels=[0, 1]).ravel()

    # Calculate Supply Recall (Non-Regular)
    supply_recall = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0

    # Calculate Supply False Positive Rate
    supply_false_positive_rate = fp_s / (fp_s + tn_s) if (fp_s + tn_s) > 0 else 0.0
    
    # Combined weighted score: higher is better
    # We combine recall (reward) and (1 - FPR) (reward for correctly classifying normal)
    combined_score = (weight_recall * supply_recall) + (weight_false_positive_rate * (1 - supply_false_positive_rate))
    
    return combined_score

# --- 1. Load the Merged DataFrame ---
print("--- Loading Merged DataFrame ---")
try:
    merged_df = pd.read_csv(INPUT_MERGED_DF_PATH)
except FileNotFoundError:
    print(f"Error: '{INPUT_MERGED_DF_PATH}' not found. Please provide the correct path to your merged_df CSV.")
    exit()

# Detect and drop any columns that start with 'Unnamed:' (e.g., 'Unnamed: 0')
unnamed_cols = [col for col in merged_df.columns if 'Unnamed:' in str(col)]
if unnamed_cols:
    print(f"Warning: Dropping the following 'Unnamed' columns: {unnamed_cols}")
    merged_df = merged_df.drop(columns=unnamed_cols)
else:
    print("No 'Unnamed' columns found in the loaded DataFrame.")

merged_df['Period_Start_Date'] = pd.to_datetime(merged_df['Period_Start_Date'])
merged_df['Period_End_Date'] = pd.to_datetime(merged_df['Period_End_Date'])

print(f"Merged DataFrame loaded successfully. Total rows: {len(merged_df)}")
print(merged_df.head())

# --- Define features and target ---
feature_columns = [col for col in merged_df.columns if col not in
                   ['Supply_ID', 'Period_Start_Date', 'Period_End_Date', 'Period_Index', 'Daily_Std_Change_Period', 'IQR_Consumption_Period', 'Max_Consumption_Period', 'Std_Consumption_Period',
                    'CLUSTER', 'Is_Non_Regular', 'anomaly_prediction_ocsvm', 'anomaly_score_ocsvm']]

X_full = merged_df[feature_columns]
y_full = merged_df['Is_Non_Regular'] # True labels for periods
supply_ids_full = merged_df['Supply_ID'] # Supply IDs for grouping

print(f"Features selected for OCSVM: {feature_columns}")
print(f"Number of features: {len(feature_columns)}")
print(f"Total non-regular periods: {y_full.sum()}")
print(f"Total regular periods: {(y_full == 0).sum()}")

# --- Initial Stratified Train-Test Split (at Supply_ID level) ---
# This ensures a truly unseen test set for final unbiased evaluation.
print("\n--- Performing Initial Stratified Train-Test Split (by Supply_ID) ---")

unique_supply_ids = merged_df['Supply_ID'].unique()
# Create labels for stratified splitting of Supply_IDs: 1 if supply has ANY non-regular period, 0 otherwise.
supply_has_non_regular = merged_df.groupby('Supply_ID')['Is_Non_Regular'].max()

# Use StratifiedShuffleSplit to create development (for CV) and test sets
# test_size=0.2 means 20% of unique supplies go to the test set
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for dev_idx, test_idx in splitter.split(unique_supply_ids, supply_has_non_regular.loc[unique_supply_ids]):
    dev_supply_ids = unique_supply_ids[dev_idx]
    test_supply_ids = unique_supply_ids[test_idx]

df_dev = merged_df[merged_df['Supply_ID'].isin(dev_supply_ids)].copy().reset_index(drop=True)
df_test = merged_df[merged_df['Supply_ID'].isin(test_supply_ids)].copy().reset_index(drop=True)

X_dev, y_dev, supply_ids_dev = df_dev[feature_columns], df_dev['Is_Non_Regular'], df_dev['Supply_ID']
X_test, y_test, supply_ids_test = df_test[feature_columns], df_test['Is_Non_Regular'], df_test['Supply_ID']

print(f"Development Set: {len(dev_supply_ids)} supplies, {len(df_dev)} periods ({y_dev.sum()} non-regular)")
print(f"Test Set: {len(test_supply_ids)} supplies, {len(df_test)} periods ({y_test.sum()} non-regular)")

# --- Feature Preparation (Imputation and Scaling) ---
# Fit Imputer and Scaler ONLY on the normal data from the development set
# These fitted objects will be reused for all folds and for the final model and test set.
print("\n--- Fitting Imputer and Scaler on Development Set Normal Data ---")
X_dev_normal_raw = X_dev[y_dev == 0]

imputer = SimpleImputer(strategy='median')
X_dev_normal_imputed = imputer.fit_transform(X_dev_normal_raw)

scaler = StandardScaler()
X_dev_normal_scaled_for_fit = scaler.fit_transform(X_dev_normal_imputed)

print("Imputer and Scaler fitted successfully on normal development data.")

# --- Stratified K-Fold Cross-Validation for Hyperparameter Tuning ---
print(f"\n--- Starting One-Class SVM Hyperparameter Tuning ({N_SPLITS}-Fold Cross-Validation) ---")

# Store results for each fold and hyperparameter combination
cv_results = []
best_avg_weighted_score_cv = -np.inf
best_cv_params = {}
best_cv_threshold = None # This will be the threshold that performs best on average across folds for the best params

# Create StratifiedKFold splitter based on Supply_IDs within the development set
# Stratify by whether a supply contains any non-regular periods
dev_supply_has_non_regular = df_dev.groupby('Supply_ID')['Is_Non_Regular'].max()
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Iterate through hyperparameters
for nu in NU_VALUES:
    for gamma in GAMMA_VALUES:
        print(f"\n  Testing nu={nu}, gamma={gamma}...")
        fold_weighted_scores = []
        
        # Iterate through folds
        for fold_n, (train_fold_idx, val_fold_idx) in enumerate(skf.split(dev_supply_ids, dev_supply_has_non_regular.loc[dev_supply_ids])):
            
            # Get Supply_IDs for the current fold
            train_fold_supply_ids = dev_supply_ids[train_fold_idx]
            val_fold_supply_ids = dev_supply_ids[val_fold_idx]

            # Filter dataframes for the current fold
            df_train_fold = df_dev[df_dev['Supply_ID'].isin(train_fold_supply_ids)].copy().reset_index(drop=True)
            df_val_fold = df_dev[df_dev['Supply_ID'].isin(val_fold_supply_ids)].copy().reset_index(drop=True)

            X_train_fold = df_train_fold[feature_columns]
            y_train_fold = df_train_fold['Is_Non_Regular']
            X_val_fold = df_val_fold[feature_columns]
            y_val_fold = df_val_fold['Is_Non_Regular']

            # --- Impute and Scale data for this fold ---
            # Apply the *already fitted* imputer and scaler (from full dev normal data)
            X_train_imputed_fold = imputer.transform(X_train_fold)
            X_val_imputed_fold = imputer.transform(X_val_fold)

            X_train_scaled_fold = scaler.transform(X_train_imputed_fold)
            X_val_scaled_fold = scaler.transform(X_val_imputed_fold)

            # --- OCSVM Training for the current fold ---
            # Train OCSVM ONLY on normal data from the current training fold
            X_train_ocsvm_fit_fold = X_train_scaled_fold[y_train_fold == 0]
            
            if X_train_ocsvm_fit_fold.shape[0] == 0: # Check if there's no normal data in this fold's training set
                print(f"    Fold {fold_n+1}: No normal data in training subset for OCSVM fit. Skipping fold.")
                fold_weighted_scores.append(-1.0) # Assign a penalty score
                continue

            ocsvm = OneClassSVM(kernel='rbf', nu=nu, gamma=gamma)
            ocsvm.fit(X_train_ocsvm_fit_fold)

            # Get decision scores for the current validation fold
            val_scores = ocsvm.decision_function(X_val_scaled_fold)
            
            # --- Threshold Optimization for this fold's validation set ---
            # Find the best threshold for this (nu, gamma) on the current fold's validation set
            
            # Objective function to minimize (negative weighted score)
            def objective(threshold, scores, true_labels, df_info):
                # OCSVM decision function: lower scores are more anomalous.
                # So, score < threshold means prediction is anomaly (1)
                predictions = (scores < threshold).astype(int) 
                return -calculate_weighted_score(true_labels, predictions, df_info, WEIGHT_RECALL, WEIGHT_FPR)

            # Use a limited, representative set of thresholds for efficiency
            unique_scores = np.sort(np.unique(val_scores))
            # Select up to 100 evenly spaced thresholds or all unique scores if fewer
            if len(unique_scores) > 100:
                threshold_candidates = np.percentile(unique_scores, np.linspace(0, 100, 100))
            else:
                threshold_candidates = unique_scores

            best_fold_score = -np.inf
            best_fold_threshold = None

            # Iterate through threshold candidates
            for t_candidate in threshold_candidates:
                current_score = -objective(t_candidate, val_scores, y_val_fold, df_val_fold[['Supply_ID', 'Is_Non_Regular']])
                if current_score > best_fold_score:
                    best_fold_score = current_score
                    best_fold_threshold = t_candidate
            
            fold_weighted_scores.append(best_fold_score)
            print(f"    Fold {fold_n+1} (nu={nu}, gamma={gamma}): Weighted Score = {best_fold_score:.4f} (Threshold={best_fold_threshold:.4f})")
        
        # Calculate average weighted score for this (nu, gamma) across all folds
        avg_score_for_params = np.mean(fold_weighted_scores)
        cv_results.append({
            'nu': nu,
            'gamma': gamma,
            'avg_weighted_score_cv': avg_score_for_params
        })

        # Update best overall parameters if current combination is better
        if avg_score_for_params > best_avg_weighted_score_cv:
            best_avg_weighted_score_cv = avg_score_for_params
            best_cv_params = {'nu': nu, 'gamma': gamma}

# Convert CV results to DataFrame and display
cv_results_df = pd.DataFrame(cv_results).sort_values(by='avg_weighted_score_cv', ascending=False).reset_index(drop=True)
print("\n--- K-Fold Cross-Validation Summary (Averaged Across Folds) ---")
print(cv_results_df.to_string())

print(f"\nBest hyperparameters from K-Fold CV: {best_cv_params}")
print(f"Average weighted score for best params: {best_avg_weighted_score_cv:.4f}")

# --- Determine the FINAL BEST THRESHOLD on the ENTIRE Development Set ---
# After finding the best hyperparameters, we train a final OCSVM on the entire normal development set
# and then find the single best threshold on the *entire* development set's decision scores.
print("\n--- Determining Final Optimal Threshold on Entire Development Set ---")

# Train final OCSVM model with best parameters found during CV
final_ocsvm_model_for_threshold = OneClassSVM(kernel='rbf', nu=best_cv_params['nu'], gamma=best_cv_params['gamma'])
final_ocsvm_model_for_threshold.fit(X_dev_normal_scaled_for_fit) # Use the scaler/imputer fitted on full dev normal data

# Get decision scores for the entire development set
dev_scores = final_ocsvm_model_for_threshold.decision_function(scaler.transform(imputer.transform(X_dev)))

# Optimize the threshold on the entire development set
unique_dev_scores = np.sort(np.unique(dev_scores))
if len(unique_dev_scores) > 100:
    threshold_candidates_dev = np.percentile(unique_dev_scores, np.linspace(0, 100, 100))
else:
    threshold_candidates_dev = unique_dev_scores

final_best_threshold = None
max_dev_weighted_score = -np.inf

for t_candidate in threshold_candidates_dev:
    current_score = -objective(t_candidate, dev_scores, y_dev, df_dev[['Supply_ID', 'Is_Non_Regular']])
    if current_score > max_dev_weighted_score:
        max_dev_weighted_score = current_score
        final_best_threshold = t_candidate

print(f"Final optimal threshold for deployment (from Development Set): {final_best_threshold:.4f}")
print(f"Weighted Score on Development Set (with final threshold): {max_dev_weighted_score:.4f}")

# --- Final Model Training (for deployment, if saving) ---
# This is mainly for saving the model. The 'final_ocsvm_model_for_threshold' already holds the best trained model.
print("\n--- Training Final OCSVM Model for Deployment/Saving ---")
final_oc_svm_model_deploy = OneClassSVM(kernel='rbf', nu=best_cv_params['nu'], gamma=best_cv_params['gamma'])
final_oc_svm_model_deploy.fit(X_dev_normal_scaled_for_fit) # Train on all normal data from dev set

# Save the trained model, imputer, and scaler
joblib.dump(final_oc_svm_model_deploy, os.path.join(MODEL_DIR, 'final_ocsvm_Kfold_model.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
joblib.dump(imputer, os.path.join(MODEL_DIR, 'imputer.pkl'))
print(f"Final OCSVM model, scaler, and imputer saved to {MODEL_DIR}")

# --- Evaluate Final Model on the UNSEEN TEST SET ---
print("\n--- Evaluating Final OCSVM Model on UNSEEN TEST SET ---")

# Apply the *already fitted* imputer and scaler (from full dev normal data) to the test set
X_test_imputed = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test_imputed)

# Get decision scores for the TEST set
test_decision_scores = final_oc_svm_model_deploy.decision_function(X_test_scaled)

# Add anomaly scores and true labels to a temporary DataFrame for easier plotting
df_test_plot = df_test.copy()
df_test_plot['anomaly_score'] = test_decision_scores
df_test_plot['is_anomaly_true'] = df_test_plot['Is_Non_Regular'] # Using 1 for non-regular (anomaly), 0 for regular (normal)

#Save test df with scores
df_test_plot.to_csv(f'{DATA_DIR}/test_df_results.csv', index=False)

# Apply the final best threshold found on the development set
test_period_predictions = (test_decision_scores < final_best_threshold).astype(int) # 1 for anomaly, 0 for normal

# Calculate final weighted score on the test set (using supply-level logic)
final_weighted_score_on_test = calculate_weighted_score(y_test, test_period_predictions, df_test[['Supply_ID', 'Is_Non_Regular']], WEIGHT_RECALL, WEIGHT_FPR)

# For detailed metric reporting on the test set, get confusion matrix components
# Ensure 'Is_Non_Regular' and 'Supply_ID' from df_test are used for grouping
temp_df_test_eval = df_test[['Supply_ID', 'Is_Non_Regular']].copy()
temp_df_test_eval['y_pred_period'] = test_period_predictions

true_non_regular_supplies_test = temp_df_test_eval.groupby('Supply_ID')['Is_Non_Regular'].max()
predicted_anomalous_supplies_test = temp_df_test_eval.groupby('Supply_ID')['y_pred_period'].max()

# Align the series for confusion matrix
all_test_supplies_ids = true_non_regular_supplies_test.index.union(predicted_anomalous_supplies_test.index)
y_true_supply_test_aligned = true_non_regular_supplies_test.reindex(all_test_supplies_ids, fill_value=0)
y_pred_supply_test_aligned = predicted_anomalous_supplies_test.reindex(all_test_supplies_ids, fill_value=0)

tn_s_test, fp_s_test, fn_s_test, tp_s_test = confusion_matrix(y_true_supply_test_aligned, y_pred_supply_test_aligned, labels=[0, 1]).ravel()

supply_recall_non_regular_test = tp_s_test / (tp_s_test + fn_s_test) if (tp_s_test + fn_s_test) > 0 else 0.0
supply_false_positive_rate_test = fp_s_test / (fp_s_test + tn_s_test) if (fp_s_test + tn_s_test) > 0 else 0.0


print(f"\n--- Final OCSVM Performance on UNSEEN TEST SET (Supply-Level) ---")
print(f"  True Positives (Supplies): {tp_s_test}")
print(f"  False Positives (Supplies): {fp_s_test}")
print(f"  False Negatives (Supplies): {fn_s_test}")
print(f"  True Negatives (Supplies): {tn_s_test}")
print(f"  Supply Recall (Non-Regular): {supply_recall_non_regular_test:.4f}")
print(f"  Supply False Positive Rate: {supply_false_positive_rate_test:.4f}")
print(f"  Final Weighted Score (Test Set): {final_weighted_score_on_test:.4f}")


# --- Visualizing Feature Distributions for Normal vs. Non-Regular Periods in Test Set ---
print("\n--- Visualizing Feature Distributions (Normal vs. Non-Regular) on Test Set ---")

# Ensure df_test has 'Is_Non_Regular' for hue
for feature in feature_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_test, x=feature, hue='Is_Non_Regular', kde=True, stat='density', common_norm=False,
                 palette={0: 'blue', 1: 'red'})
    plt.title(f'Distribution of {feature} (Test Set)')
    plt.xlabel(feature)
    plt.ylabel('Density')
    plt.legend(title='True Label', labels=['Regular', 'Non-Regular'])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, f'feature_distribution_{feature}.png'))
    plt.close() # Close the plot to prevent it from displaying multiple plots at once
    print(f"  Saved distribution plot for feature: {feature}")

print(f"All feature distribution plots saved to {FIGURE_DIR}")


# --- Feature Importance Study (Permutation Importance) ---
print("\n--- Performing Feature Importance Study (Permutation Importance) ---")

# Calculate baseline score on the *original* test set
baseline_score = final_weighted_score_on_test
print(f"Baseline Weighted Score on Test Set: {baseline_score:.4f}")

feature_importances = {}

# Use a copy of the scaled test data for permutation
X_test_scaled_copy = X_test_scaled.copy()

for i, feature_name in enumerate(feature_columns):
    print(f"  Permuting feature: {feature_name}...")
    
    # Store the original column
    original_column = X_test_scaled_copy[:, i].copy()
    
    # Shuffle the column
    np.random.shuffle(X_test_scaled_copy[:, i])
    
    # Get new decision scores with the shuffled feature
    shuffled_test_decision_scores = final_oc_svm_model_deploy.decision_function(X_test_scaled_copy)
    
    # Apply the final best threshold
    shuffled_test_period_predictions = (shuffled_test_decision_scores < final_best_threshold).astype(int)
    
    # Calculate weighted score with shuffled feature
    shuffled_score = calculate_weighted_score(y_test, shuffled_test_period_predictions, df_test[['Supply_ID', 'Is_Non_Regular']], WEIGHT_RECALL, WEIGHT_FPR)
    
    # Calculate importance (drop in score)
    importance = baseline_score - shuffled_score
    feature_importances[feature_name] = importance
    
    # Restore the original column for the next iteration
    X_test_scaled_copy[:, i] = original_column

# Sort features by importance
sorted_importances = sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

print("\n--- Feature Importance Results (Permutation Importance on Test Set) ---")
for feature, importance in sorted_importances:
    print(f"  {feature}: {importance:.4f}")

# --- Visualize Feature Importance ---
features_imp = [item[0] for item in sorted_importances]
importances_imp = [item[1] for item in sorted_importances]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_imp, y=features_imp, palette='viridis')
plt.xlabel("Importance (Decrease in Weighted Score)")
plt.ylabel("Feature")
plt.title("One-Class SVM Permutation Feature Importance (Test Set)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURE_DIR, 'feature_importance.png'))
plt.show()
print(f"Feature importance plot saved to {os.path.join(FIGURE_DIR, 'feature_importance.png')}")


# --- Add final predictions and scores to the original merged_df for full analysis ---
# Use the final deployed model, imputer, and scaler to predict on the *entire* original dataset.
print("\n--- Applying Final Model Predictions to Entire Original Dataset ---")
X_full_imputed = imputer.transform(X_full)
X_full_scaled = scaler.transform(X_full_imputed)

merged_df['anomaly_score_ocsvm'] = final_oc_svm_model_deploy.decision_function(X_full_scaled)
# Apply the best_overall_threshold to the entire dataset for consistent binary flags
merged_df['anomaly_prediction_ocsvm_thresholded'] = np.where(
    merged_df['anomaly_score_ocsvm'] < final_best_threshold, -1, 1
)

print("\n--- Final Merged DataFrame Head with OCSVM Results (including thresholded predictions) ---")
print(merged_df[['Supply_ID', 'Period_Start_Date', 'Period_End_Date',
                 'Mean_Consumption_Period', 'Is_Non_Regular',
                 'anomaly_score_ocsvm', 'anomaly_prediction_ocsvm_thresholded']].head())

print("\n--- Counts of OCSVM Anomaly Predictions (Period-Level, after thresholding) ---")
print(merged_df['anomaly_prediction_ocsvm_thresholded'].value_counts())

#Save the updated merged_df with OCSVM predictions
merged_df.to_csv(OUTPUT_FINAL_CSV_PATH, index=False)
print(f"\nUpdated merged_df with OCSVM predictions saved to {OUTPUT_FINAL_CSV_PATH}")


