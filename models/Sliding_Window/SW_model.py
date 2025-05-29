import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#Paths
sw_df_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Sliding_Window/sliding_window_dataset.csv'
labels_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Provided_data/Labels.csv'

#Function ot combine features and labels (necessary since labels dataset not processed yet)
def label_dataset(df_feature, df_label):

    # Merge the features dataset with the labels dataset on Supply_ID
    labeled_features_df = pd.merge(df_feature, df_label, on='Supply_ID', how='inner')
    labeled_features_df = labeled_features_df.rename(columns={'CLUSTER': 'Label'})
    
    # Encode necessary values
    label_mapping = {'Regolare': 0, 'Anomalia': 1, 'Frode': 1}
    labeled_features_df['Label'] = labeled_features_df['Label'].map(label_mapping)

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in labeled_features_df.columns:
        labeled_features_df = labeled_features_df.drop(columns=['Unnamed: 0', 'Supply_ID'])

    return labeled_features_df

if __name__ == '__main__':

    #Feature and label datasets
    sw_features = pd.read_csv(sw_df_path)  
    sw_labels = pd.read_csv(labels_path, encoding='utf-16', sep='\t', decimal=',')

    #Combine features and labels
    sw_df = label_dataset(sw_features, sw_labels)

    #Split data

    # Split the dataset into features (X) and labels (y)
    X = sw_df.drop(columns=['Label'])  # Features
    y = sw_df['Label']  # Labels

    # Perform stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Vizualize final split shapes
    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    print(y_train.value_counts())
    print(y_test.value_counts())

    # Calculate scale_pos_weight
    negative_count = y_train.value_counts()[0]
    positive_count = y_train.value_counts()[1]
    scale_pos_weight = negative_count / positive_count
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")


    # Initialize the XGBoost classifier
    model = XGBClassifier(
        objective='binary:logistic',  # For binary classification
        max_depth=6,                  # Maximum depth of trees
        learning_rate=0.1,            # Learning rate (eta)
        n_estimators=100,             # Number of boosting rounds
        random_state=42,               # Random seed for reproducibility
        scale_pos_weight = scale_pos_weight
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
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Regular', 'Fraud'], yticklabels=['Regular', 'Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix Heatmap')

    plt.savefig('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Sliding_Window/model_heatmap.png')
    plt.show()


    