import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

print(df.shape)
df.head()

#check datatypes
print(df.dtypes)

# Convert to numeric type and coerce errors to NaN(Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True) #remove rows with NaN values

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

# Define features and target variable
X = df.drop(columns=['Churn'])
y = df['Churn']

print(df.head())
print(">>>>info<<<<")
print(df.info())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=45)

# Standardize(Scale) the data to allow MLP to converge
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_test)
print(X_train)

#create instance of MLP with 3 hidden layers and 500 iterations
mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10), max_iter=500)

#fit the scaled data
mlp.fit(X_train, y_train)

# Perform prediction and generate confusion matrix
pred_result = mlp.predict(X_test)
cm = confusion_matrix(y_test, pred_result)
print(cm)

# Generating the classification report
c_report = classification_report(y_test, pred_result)
print(c_report)

