import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

#Read dataset into pandas dataframe
input_path = '/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Basic_model/Data/Basic_model_dataset_quarterhouravg_400days.csv'
dataset = pd.read_csv(input_path)

#Extract imbalanced labels (non regular)
rows_to_include_train = list(range(6))
rows_to_include_test = list(range(6,10))
train_balance = dataset.loc[rows_to_include_train, :]
test_balance = dataset.loc[rows_to_include_test, :]

#Split only regulars
to_split = dataset.loc[10:, :]
X = to_split.drop(columns=['is_Regular', 'Supply_ID'])
Y = to_split['is_Regular']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#Aggreagate training and test with balance

X_train = pd.concat([X_train, train_balance.drop(columns=['is_Regular', 'Supply_ID'])], axis=0)
X_test = pd.concat([X_test, test_balance.drop(columns=['is_Regular', 'Supply_ID'])], axis=0)

Y_train = pd.concat([Y_train, train_balance['is_Regular']], axis=0)
Y_test = pd.concat([Y_test, test_balance['is_Regular']], axis=0)


# Reset indices to ensure consistency
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
Y_train.reset_index(drop=True, inplace=True)
Y_test.reset_index(drop=True, inplace=True)


#Ratio to handle imbalance
ratio = len(Y_train[Y_train == 0]) / len(Y_train[Y_train == 1])

#Printing 

print(X_train.sample(10))
print(Y_train.sample(10))
print(X_test.sample(10))
print(Y_test.sample(10))


#Create model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1, 
    random_state=42,
    scale_pos_weight=ratio
)

#Train model
model.fit(X_train, Y_train)

#Run on test set
y_pred = model.predict(X_test)

#Compute probabilities for AUC
y_prob = model.predict_proba(X_test)[:, 0]

#Evaluate model
accuracy = accuracy_score(Y_test, y_pred)
print(f'Model accuracy: {accuracy}')

print("Classification Report:")
print(classification_report(Y_test, y_pred))

# Generate the confusion matrix
conf_matrix = confusion_matrix(Y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Regular', 'Regular'], yticklabels=['Non-Regular', 'Regular'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')

#Save confusion matrix heatmap
plt.savefig('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Basic_model/Plots/Confusion_matrix(quarterhour_400).png')


#Compute ROC curve
FPR, TPR, thresholds = roc_curve(Y_test, y_prob, pos_label=0) # Specify the positive label

# Calculate AUC-score
auc_score = roc_auc_score(Y_test, y_prob)
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

#Save ROC curve
plt.savefig('/Users/diegozago2312/Documents/Work/Ennel_Innothon/Challenge2/Basic_model/Plots/ROC_curve_(quarterhour_400).png')


plt.show()

