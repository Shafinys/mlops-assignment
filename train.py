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
import optuna
import joblib

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def train_random_forest(params):
    with mlflow.start_run(nested=True):
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
            input_example=X_train[:5]
        )
        
        return rf_clf, rf_accuracy

def train_logistic_regression(params):
    with mlflow.start_run(nested=True):
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
            input_example=X_train[:5]
        )
        
        return lr_clf, lr_accuracy

def objective_rf(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 10, 200),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "random_state": 42
    }
    _, accuracy = train_random_forest(params)
    return accuracy

def objective_lr(trial):
    params = {
        "C": trial.suggest_loguniform("C", 1e-5, 1e5),
        "solver": trial.suggest_categorical("solver", ["newton-cg", "lbfgs", "liblinear"]),
        "max_iter": 1000,
        "random_state": 42
    }
    _, accuracy = train_logistic_regression(params)
    return accuracy

if __name__ == "__main__":
    # Optuna study for Random Forest
    with mlflow.start_run(run_name="Random Forest Optimization"):
        study_rf = optuna.create_study(direction="maximize")
        study_rf.optimize(objective_rf, n_trials=20)
        
        print("Best Random Forest hyperparameters:", study_rf.best_params)
        
        # Train final Random Forest model with best params
        rf_model, rf_accuracy = train_random_forest(study_rf.best_params)
        
        # Save the best Random Forest model
        joblib.dump(rf_model, 'best_rf_model.joblib')
        
        print("\nRandom Forest Classifier (Best Parameters):")
        print(f"Accuracy: {rf_accuracy:.2f}")
    
    # Optuna study for Logistic Regression
    with mlflow.start_run(run_name="Logistic Regression Optimization"):
        study_lr = optuna.create_study(direction="maximize")
        study_lr.optimize(objective_lr, n_trials=20)
        
        print("Best Logistic Regression hyperparameters:", study_lr.best_params)
        
        # Train final Logistic Regression model with best params
        lr_model, lr_accuracy = train_logistic_regression(study_lr.best_params)
        
        # Save the best Logistic Regression model
        joblib.dump(lr_model, 'best_lr_model.joblib')
        
        print("\nLogistic Regression (Best Parameters):")
        print(f"Accuracy: {lr_accuracy:.2f}")
    
    # Compare models
    print("\nModel Comparison:")
    if rf_accuracy > lr_accuracy:
        print("Random Forest Classifier performed better.")
        best_model = rf_model
    elif lr_accuracy > rf_accuracy:
        print("Logistic Regression performed better.")
        best_model = lr_model
    else:
        print("Both models performed equally.")
        best_model = rf_model  # Choose Random Forest arbitrarily in case of a tie

    # Save the overall best model
    joblib.dump(best_model, 'best_model.joblib')
    print("\nBest model saved as 'best_model.joblib'")