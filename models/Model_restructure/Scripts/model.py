import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

#Datasets path
features_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Model_restructure/Data/Merged_consumption_customer.csv'
labels_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Labels.csv'

#Read datasets into dataframes
features = pd.read_csv(features_path)
labels = pd.read_csv(labels_path, encoding='utf-16', sep='\t', decimal=',')


#Function ot combine features and labels (necessary since labels dataset not processed yet)
def label_dataset(df_feature, df_label):

    # Merge the features dataset with the labels dataset on Supply_ID
    labeled_features_df = pd.merge(df_feature, df_label, on='Supply_ID', how='left')

    # Handle missing labels (if any Supply_ID in features is not in labels)
    labeled_features_df['CLUSTER'].fillna('Unknown', inplace=True)

    return labeled_features_df

#Function to pre process data
def pre_process(df):
    # Study NaN values
    nan_per_column = df.isna().sum()
    print(f'\nNaN values per column:\n{nan_per_column}')

    non_nan_per_column = (~df.isna()).sum()
    print(f'\nNon NaN values per column:\n{non_nan_per_column}')

    # Encode necessary values
    label_mapping = {'Regolare': 0, 'Anomalia': 1, 'Frode': 1}
    df['CLUSTER'] = df['CLUSTER'].map(label_mapping)

    # Handle unmapped values in CLUSTER
    if df['CLUSTER'].isna().any():
        print("Warning: Some values in the CLUSTER column were not mapped and replaced with NaN.")

    # Encode years per supply
    if 'Year' in df.columns and pd.api.types.is_numeric_dtype(df['Year']):
        df['Year_encoded'] = df.groupby('Supply_ID')['Year'].transform(lambda x: x.rank(method='dense').astype(int) - 1)
    else:
        raise ValueError("Year column is missing or not numerical!")

    #Encode supply_ID
    df['Supply_ID'] = df['Supply_ID'].str[8:].astype(int)

    
    #Dropping rows and handling NaNs

    # Drop unencoded Status column
    df = df.drop(columns=['Status'])
    print(f'\nstuts column check:\n{df.columns}\n')

    # Fill NaN values for numerical columns only
    for i in df.select_dtypes(include=['number']).columns:
        mean_b = df[i].mean()
        df[i].fillna(mean_b, inplace=True)
    
    # Drop rows that contain too many zeroes
    zero_counts = (df == 0).sum(axis=1)
    df = df[zero_counts <= 3]

    #Correctly order columns
    new_order = ['Supply_ID', 'Year_encoded', 'val_mean', 'val_max', 'val_min', 'val_std',
       'mean_available_power', 'Status_encoded', 'CLUSTER']
    df = df[new_order]

    #Rename columns
    df.rename(columns={'Year_encoded': 'periods', 'val_mean': 'consumption_mean', 'val_max': 'consumption_max', 'val_min': 'consumption_min', 'val_std': 'consumption_std', 'Status_encoded': 'status', 'CLUSTER': 'label'}, inplace=True)

    print(f'\nDataset after preprocessing:\n{df.head(15)}')

    return df



#Run code
if __name__ == '__main__':

    dataset_raw = label_dataset(features, labels)
    dataset = pre_process(dataset_raw)


    #Split data

    # Split the dataset into features (X) and labels (y)
    X = dataset.drop(columns=['label'])  # Features
    y = dataset['label']  # Labels

    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Vizualize final split shapes
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    #Vi
    print(y_train.value_counts())
    print(y_test.value_counts())

    # Initialize the XGBoost classifier
    model = XGBClassifier(
        objective='binary:logistic',  # For binary classification
        max_depth=6,                  # Maximum depth of trees
        learning_rate=0.1,            # Learning rate (eta)
        n_estimators=100,             # Number of boosting rounds
        random_state=42               # Random seed for reproducibility
    )

        # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Regular', 'Regular'], yticklabels=['Non-Regular', 'Regular'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')


    #Compute probabilities for AUC
    y_prob = model.predict_proba(X_test)[:, 0]

    #Compute ROC curve
    FPR, TPR, thresholds = roc_curve(y_test, y_prob, pos_label=1) # Specify the positive label

    # Calculate AUC-score
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"AUC Score: {auc_score}")

    #Plot the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(FPR, TPR, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()
