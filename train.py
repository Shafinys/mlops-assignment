# train.py

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train a Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Create and train a Logistic Regression model
lr_clf = LogisticRegression(random_state=42, max_iter=200)
lr_clf.fit(X_train, y_train)

# Make predictions on the test set
rf_pred = rf_clf.predict(X_test)
lr_pred = lr_clf.predict(X_test)

# Calculate accuracy for both models
rf_accuracy = accuracy_score(y_test, rf_pred)
lr_accuracy = accuracy_score(y_test, lr_pred)

# Print results
print("Random Forest Classifier:")
print(f"Accuracy: {rf_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=iris.target_names))

print("\nLogistic Regression:")
print(f"Accuracy: {lr_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=iris.target_names))

# Compare models
print("\nModel Comparison:")
if rf_accuracy > lr_accuracy:
    print("Random Forest Classifier performed better.")
elif lr_accuracy > rf_accuracy:
    print("Logistic Regression performed better.")
else:
    print("Both models performed equally.")