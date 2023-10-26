from time import time
from keras.datasets import mnist
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# load the dataset
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

# SVM
startTime = time()
svm = SVC(kernel='rbf', C=0.01, gamma=100)
svm.fit(X_train, y_train)
fitTime = time() - startTime
print(f"Fit time of SVM is: {fitTime}s")
startTimePredict = time()
y_pred = svm.predict(X_test)
predictTime = time() - startTimePredict
print(f"Predict time of SVM is: {predictTime}s")
predicted = svm.predict(X_train)
print("accuracy of test set:", metrics.accuracy_score(y_test, y_pred))
print("accuracy of train set:", metrics.accuracy_score(y_train, predicted))


