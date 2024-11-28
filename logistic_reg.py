import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# Display the first few rows and data info
print(df.head())
print(">>>>count<<<<")
print(df.count())
print(">>>>info<<<<")
print(df.info())

# Handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Encode categorical variables
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
df['PhoneService'] = df['PhoneService'].map({'Yes': 1, 'No': 0})
df['PaperlessBilling'] = df['PaperlessBilling'].map({'Yes': 1, 'No': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# One-hot encode other categorical columns
df = pd.get_dummies(df, columns=['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'])

# Drop columns that won't be used
df.drop(columns=['customerID'], inplace=True)

# Split the data for train and test
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Create LR instance and fit
lrmodel = LogisticRegression(max_iter=1000)
lrmodel.fit(X_train, y_train)

# Perform prediction and generate confusion matrix
pred_result = lrmodel.predict(X_test)
cm = confusion_matrix(y_test, pred_result)
print(cm)

# Generating the classification report
c_report = classification_report(y_test, pred_result)
print(c_report)

# Plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True)
plt.show()
