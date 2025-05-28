import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 
import os

input_path = 'models/Unsupervized/Data/unsupervized_dataset.csv'
period_features_df = pd.read_csv(input_path)
labels_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Labels.csv'


' ========= Dropping Unnamed 0 =========='
# Load your final feature dataset
# If period_features_df is already in memory, you can skip this line:
# period_features_df = pd.read_csv('period_features.csv') # Adjust filename if different

print("Original columns:", period_features_df.columns.tolist())

# Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in period_features_df.columns:
    period_features_df = period_features_df.drop(columns=['Unnamed: 0'])
    print("Dropped 'Unnamed: 0' column.")

print("Columns after dropping 'Unnamed: 0':", period_features_df.columns.tolist())

'================== Pre-process ======================'

try:
    fraud_supply_labels_df = pd.read_csv(labels_path, encoding='utf-16', sep='\t', decimal=',')

except FileNotFoundError:
    print(f"Error: '{labels_path}' not found. Please provide the correct path to your labels file.")
    exit()

# Ensure Supply_ID in labels is consistent (uppercase, stripped spaces)
fraud_supply_labels_df['Supply_ID'] = fraud_supply_labels_df['Supply_ID'].astype(str).str.strip().str.upper()

# Create 'Is_Non_Regular' combined label
# 1 for 'Anomalous' or 'Fraud', 0 for 'Regular'
fraud_supply_labels_df['Is_Non_Regular'] = fraud_supply_labels_df['CLUSTER'].apply(
    lambda x: 1 if x in ['Anomalia', 'Frode'] else 0
)

print("Labeled Data Head (with Is_Non_Regular):")
print(fraud_supply_labels_df.head())
print(f"Counts of 'CLUSTER':\n{fraud_supply_labels_df['CLUSTER'].value_counts()}")
print(f"Counts of 'Is_Non_Regular':\n{fraud_supply_labels_df['Is_Non_Regular'].value_counts()}")


# Merge period features with the new 'Is_Non_Regular' label
merged_df = pd.merge(period_features_df, fraud_supply_labels_df[['Supply_ID', 'Is_Non_Regular']], on='Supply_ID', how='left')

# Handle cases where some supplies in period_features_df might not have labels (fill with 0/Regular)
merged_df['Is_Non_Regular'] = merged_df['Is_Non_Regular'].fillna(0).astype(int)

print(f"\nMerged DataFrame head with Is_Non_Regular label:")
print(merged_df[['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular']].head())
print(f"Total rows in merged_df: {len(merged_df)}")
print(f"Total periods from 'Is_Non_Regular = 1' supplies: {merged_df[merged_df['Is_Non_Regular'] == 1].shape[0]}")
print(f"Total periods from 'Is_Non_Regular = 0' supplies: {merged_df[merged_df['Is_Non_Regular'] == 0].shape[0]}")

# # Save final merged_df
# merged_df.to_csv('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/Data/OCSVM_anomaly_detection.csv')
# print(f'Final merged df saved')


'============== Hyperparameter tuning ============'

print("\n--- Starting One-Class SVM Hyperparameter Tuning ---")

# Define features for OCSVM (exclude identifiers and target labels)
feature_columns = [col for col in merged_df.columns if col not in
                   ['Supply_ID', 'Period_Start_Date', 'Period_End_Date', 'Period_Index', 'CLUSTER', 'Is_Non_Regular']]

X = merged_df[feature_columns]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a range of hyperparameters to test
nu_values = [0.005, 0.01, 0.02, 0.05, 0.1]
gamma_values = ['scale', 0.1, 0.01, 0.001] # Test 'scale' first, then specific values

best_score = -np.inf # Optimize for (Recall - False Alarm Rate)
best_params = {}
evaluation_results = []

for nu_val in nu_values:
    for gamma_val in gamma_values:
        print(f"\n--- Testing nu={nu_val}, gamma={gamma_val} ---")
        oc_svm_model = OneClassSVM(kernel='rbf', nu=nu_val, gamma=gamma_val)
        oc_svm_model.fit(X_scaled)

        predictions = oc_svm_model.predict(X_scaled)

        # Separate indices for 'Non-Regular' and 'Regular' supplies
        non_regular_supply_periods_indices = merged_df[merged_df['Is_Non_Regular'] == 1].index
        regular_supply_periods_indices = merged_df[merged_df['Is_Non_Regular'] == 0].index

        # OCSVM predictions subset for evaluation
        pred_on_non_regular_supplies = predictions[non_regular_supply_periods_indices]
        pred_on_regular_supplies = predictions[regular_supply_periods_indices]

        # Calculate Recall for Non-Regular Supplies (our "target" events)
        if len(non_regular_supply_periods_indices) > 0:
            recall_non_regular_supplies = (pred_on_non_regular_supplies == -1).sum() / len(non_regular_supply_periods_indices)
        else:
            recall_non_regular_supplies = 0.0 # No non-regular supplies to recall

        # Calculate False Alarm Rate (anomalies flagged in truly regular supplies)
        if len(regular_supply_periods_indices) > 0:
            false_alarm_rate = (pred_on_regular_supplies == -1).sum() / len(regular_supply_periods_indices)
        else:
            false_alarm_rate = 0.0 # No regular supplies to create false alarms

        total_anomalies_flagged = (predictions == -1).sum()
        total_periods = len(predictions)
        overall_anomaly_rate = (total_anomalies_flagged / total_periods) * 100

        evaluation_results.append({
            'nu': nu_val,
            'gamma': gamma_val,
            'total_anomalies_flagged': total_anomalies_flagged,
            'overall_anomaly_rate_%': overall_anomaly_rate,
            'recall_non_regular_supplies': recall_non_regular_supplies,
            'false_alarm_rate': false_alarm_rate
        })
        
        print(f"  Overall Anomaly Rate: {overall_anomaly_rate:.2f}%")
        print(f"  Recall for Non-Regular Supplies: {recall_non_regular_supplies:.3f}")
        print(f"  False Alarm Rate (on Regular Supplies): {false_alarm_rate:.3f}")

        # Optimization metric: Maximize recall while penalizing false alarms
        current_score = recall_non_regular_supplies - false_alarm_rate
        if current_score > best_score:
            best_score = current_score
            best_params = {'nu': nu_val, 'gamma': gamma_val}

results_df_eval = pd.DataFrame(evaluation_results)
print("\n--- Hyperparameter Evaluation Results ---")
print(results_df_eval.sort_values(by='recall_non_regular_supplies', ascending=False))

print(f"\nBest parameters found (based on Recall - False Alarm Rate): {best_params}")


# '============== Vizualize tuning results ============='
# plt.figure(figsize=(16, 7))

# plt.subplot(1, 2, 1)
# sns.lineplot(data=results_df_eval, x='nu', y='recall_non_regular_supplies', hue='gamma', marker='o')
# plt.title('Recall for Non-Regular Supplies vs. Nu')
# plt.xlabel('nu (Expected Anomaly Proportion)')
# plt.ylabel('Recall on Non-Regular Supplies')
# plt.grid(True)
# plt.xscale('log') # Use log scale for nu if values span orders of magnitude

# plt.subplot(1, 2, 2)
# sns.lineplot(data=results_df_eval, x='nu', y='false_alarm_rate', hue='gamma', marker='o')
# plt.title('False Alarm Rate vs. Nu')
# plt.xlabel('nu (Expected Anomaly Proportion)')
# plt.ylabel('False Alarm Rate')
# plt.grid(True)
# plt.xscale('log') # Use log scale for nu
# plt.tight_layout()
# plt.show()


'============ Final training with selected paramerters ==========='

print(f"\n--- Training Final OCSVM Model with nu={best_params['nu']}, gamma={best_params['gamma']} ---")

final_oc_svm_model = OneClassSVM(kernel='rbf', nu=best_params['nu'], gamma=best_params['gamma'])
final_oc_svm_model.fit(X_scaled)

# Add predictions and scores to the merged DataFrame
merged_df['anomaly_prediction_ocsvm'] = final_oc_svm_model.predict(X_scaled)
merged_df['anomaly_score_ocsvm'] = final_oc_svm_model.decision_function(X_scaled)

print("\n--- Final Merged DataFrame Head with OCSVM Results ---")
print(merged_df[['Supply_ID', 'Period_Start_Date', 'Period_End_Date',
                 'Mean_Consumption_Period', 'Is_Non_Regular',
                 'anomaly_prediction_ocsvm', 'anomaly_score_ocsvm']].head())

print("\n--- Counts of OCSVM Anomaly Predictions ---")
print(merged_df['anomaly_prediction_ocsvm'].value_counts())

# --- Cross-checking OCSVM Anomalies with 'Is_Non_Regular' Label ---
print("\n--- Cross-Checking OCSVM Anomalies with Labeled Data ---")

# Periods flagged as anomalous by OCSVM
ocsvm_anomalies = merged_df[merged_df['anomaly_prediction_ocsvm'] == -1]

# How many of OCSVM's anomalies are from 'Is_Non_Regular = 1' supplies (desirable hits)
ocsvm_anomalies_from_non_regular_supplies = ocsvm_anomalies[ocsvm_anomalies['Is_Non_Regular'] == 1]
print(f"Number of OCSVM Anomalies from 'Non-Regular' supplies: {len(ocsvm_anomalies_from_non_regular_supplies)}")

# How many of OCSVM's anomalies are from 'Is_Non_Regular = 0' supplies (false alarms from labeling perspective)
ocsvm_anomalies_from_regular_supplies = ocsvm_anomalies[ocsvm_anomalies['Is_Non_Regular'] == 0]
print(f"Number of OCSVM Anomalies from 'Regular' supplies: {len(ocsvm_anomalies_from_regular_supplies)}")

# How many 'Is_Non_Regular = 1' periods did OCSVM *miss* (false negatives)
non_regular_periods_total = merged_df[merged_df['Is_Non_Regular'] == 1]
ocsvm_missed_non_regular_periods = non_regular_periods_total[non_regular_periods_total['anomaly_prediction_ocsvm'] == 1]
print(f"Number of 'Non-Regular' periods missed by OCSVM: {len(ocsvm_missed_non_regular_periods)}")

# Top 10 anomalies (most likely anomaly)
print("\n--- Top 10 OCSVM Anomalies (lowest scores) ---")
print(merged_df.sort_values(by='anomaly_score_ocsvm').head(10)[
    ['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular', 'anomaly_score_ocsvm']
])

# Top 10 anomalies part of a fraud supply
print("\n--- Top 10 OCSVM Anomalies from 'Non-Regular' Supplies ---")
print(ocsvm_anomalies_from_non_regular_supplies.sort_values(by='anomaly_score_ocsvm').head(10)[
    ['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular', 'anomaly_score_ocsvm']
])

# Top 10 anomalies from regular supplies
print("\n--- Top 10 OCSVM Anomalies from 'Regular' Supplies (potential false alarms) ---")
print(ocsvm_anomalies_from_regular_supplies.sort_values(by='anomaly_score_ocsvm').head(10)[
    ['Supply_ID', 'Period_Start_Date', 'Mean_Consumption_Period', 'Is_Non_Regular', 'anomaly_score_ocsvm']
])


'========== Save model and scaler =========='

print("\n--- Saving the Final OCSVM Model and Scaler ---")

#Define models directory
model_dir = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/models/Unsupervized/models'

model_filename = os.path.join(model_dir, 'final_ocsvm_model.joblib')
scaler_filename = os.path.join(model_dir, 'scaler.joblib')

# Save the OCSVM model
joblib.dump(final_oc_svm_model, model_filename)
print(f"OCSVM model saved to: {model_filename}")

#Save the OCSVM scaler
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to: {scaler_filename}")

print(f'\n Script done')

