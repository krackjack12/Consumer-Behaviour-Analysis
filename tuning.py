import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
data = pd.read_csv('/Users/krishjoshi/Downloads/CustomerBehaviour.csv')

X = data.drop(columns=['Timestamp'])  # Features
y = data['Shopping_Satisfaction']  # Target variable

# Encode categorical variables
X = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'max_iter': [100, 1000, 10000]  # Maximum number of iterations
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-fold cross-validation
    verbose=1,
    n_jobs=-1  # Use all available CPU cores for parallel processing
)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Create a Logistic Regression model with the best hyperparameters
logistic_regression_model = LogisticRegression(**best_params, random_state=42)

# Train the model with the best hyperparameters
logistic_regression_model.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = logistic_regression_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=1)

print("Final Logistic Regression Model:")
print(f'Accuracy: {accuracy:.2f}')
print(report)