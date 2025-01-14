# train.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_random_forest(params):
    with mlflow.start_run(run_name="Random Forest"):
        # Log parameters
        mlflow.log_params(params)
        
        # Create and train a Random Forest Classifier
        rf_clf = RandomForestClassifier(**params)
        rf_clf.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        rf_pred = rf_clf.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", rf_accuracy)
        
        # Infer the model signature
        signature = infer_signature(X_train, rf_clf.predict(X_train))
        
        # Log the trained model with signature and input example
        mlflow.sklearn.log_model(
            rf_clf, 
            "random_forest_model",
            signature=signature,
            input_example=X_train[:5]  # Use the first 5 samples as an example
        )
        
        # Print results
        print("Random Forest Classifier:")
        print(f"Accuracy: {rf_accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, rf_pred, target_names=iris.target_names))
        
        return rf_accuracy

def train_logistic_regression(params):
    with mlflow.start_run(run_name="Logistic Regression"):
        # Log parameters
        mlflow.log_params(params)
        
        # Create and train a Logistic Regression model
        lr_clf = LogisticRegression(**params)
        lr_clf.fit(X_train, y_train)
        
        # Make predictions and calculate accuracy
        lr_pred = lr_clf.predict(X_test)
        lr_accuracy = accuracy_score(y_test, lr_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", lr_accuracy)
        
        # Infer the model signature
        signature = infer_signature(X_train, lr_clf.predict(X_train))
        
        # Log the trained model with signature and input example
        mlflow.sklearn.log_model(
            lr_clf, 
            "logistic_regression_model",
            signature=signature,
            input_example=X_train[:5]  # Use the first 5 samples as an example
        )
        
        # Print results
        print("\nLogistic Regression:")
        print(f"Accuracy: {lr_accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, lr_pred, target_names=iris.target_names))
        
        return lr_accuracy

if __name__ == "__main__":
    # Random Forest parameters
    rf_params = {
        "n_estimators": 100,
        "random_state": 42
    }
    
    # Logistic Regression parameters
    lr_params = {
        "random_state": 42,
        "max_iter": 200
    }
    
    # Train models and get accuracies
    rf_accuracy = train_random_forest(rf_params)
    lr_accuracy = train_logistic_regression(lr_params)
    
    # Compare models
    print("\nModel Comparison:")
    if rf_accuracy > lr_accuracy:
        print("Random Forest Classifier performed better.")
    elif lr_accuracy > rf_accuracy:
        print("Logistic Regression performed better.")
    else:
        print("Both models performed equally.")