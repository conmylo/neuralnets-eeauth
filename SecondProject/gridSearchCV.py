import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load the dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(f"Data Loaded")

X_train = X_train.astype(float) / 255.0
X_test = X_test.astype(float) / 255.0

# reshape dataset
X_train = X_train.reshape((X_train.shape[0], -1), order="F")
X_test = X_test.reshape((X_test.shape[0], -1), order="F")
print(f"Data Reshaped")

# standardize dataset
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(f"Data Standardized")

# change labels to even 0 and odd 1
y_train = y_train % 2
y_test = y_test % 2

# PCA
pca = PCA(.90)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
print(f"PCA is complete with {pca.n_components_} components")

# SVM hyperparameters for tuning
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "gamma": ['auto', 'scale', 0.1, 1, 10],
    "kernel": ["rbf", "poly", "linear", "sigmoid"]
}

# perfrom grid search cv
grid = GridSearchCV(SVC(), param_grid, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# show results of grid search
print(f"Best parameters set found: {grid.best_params}")
print("Grid scores:")
means = grid.cv_results_["mean_test_score"]
stds = grid.cv_results_["std_test_score"]
means_fit_time = grid.cv_results_["mean_fit_time"]
stds_fit_time = grid.cv_results_["std_fit_time"]
for mean, std, mean_fit_time, std_fit_time, params in zip(
    means,
    stds,
    means_fit_time,
    stds_fit_time,
    grid.cv_results_["params"],
):
    print(
        "%0.3f (+/-%0.03f) for %r in %0.3f (+/-%0.03f)"
        % (mean, std * 2, params, mean_fit_time, std_fit_time * 2)
    )
