# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, confusion_matrix
# import pickle

# # Load Pima Indian Diabetes dataset
# df = pd.read_csv('diabetes.csv')

# # Prepare features (X) and labels (y)
# X = df.drop('Outcome', axis=1)
# y = df['Outcome']

# # Apply Polynomial Feature Transformation
# poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
# X_poly = poly.fit_transform(X)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# # Scale the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Reduced hyperparameter grid for faster tuning
# param_grid = {
#     'C': [0.1, 1, 10],  # Fewer values for faster search
#     'gamma': [0.1, 0.01],
#     'kernel': ['linear', 'rbf']  # Reduced to two kernels
# }

# # Initialize SVM model with maximum iterations
# svm_model = SVC(probability=True, class_weight='balanced', max_iter=1000)  # Limited iterations

# # Perform Grid Search to find best hyperparameters with 3-fold cross-validation
# grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=1, cv=3)  # Reduced verbosity and folds
# grid_search.fit(X_train, y_train)

# # Get the best model and hyperparameters
# best_svm_model = grid_search.best_estimator_
# print("Best parameters found: ", grid_search.best_params_)
# print("Best cross-validation score: ", grid_search.best_score_)

# # Evaluate the model on test data
# y_pred = best_svm_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Test set accuracy: {accuracy}")

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# print("Confusion Matrix:\n", cm)

# # Save the model to a pickle file
# with open('svm_model.pkl', 'wb') as file:
#     pickle.dump(best_svm_model, file)

# # Save the scaler for future use
# with open('scaler.pkl', 'wb') as file:
#     pickle.dump(scaler, file)

# # Save the polynomial feature transformer
# with open('diabetes_svm_model.pkl', 'wb') as file:
#     pickle.dump(poly, file)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load Pima Indian Diabetes dataset
df = pd.read_csv('diabetes.csv')

# Prepare features (X) and labels (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Apply Polynomial Feature Transformation
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

# Initialize SVM model
svm_model = SVC(probability=True, class_weight='balanced')

# Perform Grid Search to find best hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=3, cv=5)
grid_search.fit(X_train, y_train)

# Get the best model and hyperparameters
best_svm_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Evaluate the model on test data
y_pred = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy: {accuracy}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Save the model, scaler, and polynomial transformer to pickle files
with open('diabetes_svm_model.pkl', 'wb') as model_file:
    pickle.dump(best_svm_model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

with open('poly_transformer.pkl', 'wb') as poly_file:
    pickle.dump(poly, poly_file)
