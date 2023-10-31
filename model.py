import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
# Assuming your dataset is loaded into a DataFrame called 'data'
data = pd.read_csv('/Users/krishjoshi/Downloads/CustomerBehaviour.csv')

X = data.drop(columns=['Timestamp'])  # Features
y = data['Shopping_Satisfaction']  # Target variable

# Encode categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Increase max_iter parameter
logistic_regression_model = LogisticRegression(max_iter=1000, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
svm_model = SVC(random_state=42)

random_forest_model.fit(X_train, y_train)
logistic_regression_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)

# Make predictions and evaluate each model
def evaluate_model(model, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=1)
    print(f'{model_name} Model:')
    print(f'Accuracy: {accuracy:.2f}')
    print(report)

evaluate_model(random_forest_model, "Random Forest")
evaluate_model(logistic_regression_model, "Logistic Regression")
evaluate_model(svm_model, "SVM")