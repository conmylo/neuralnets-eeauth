# import the libraries
from keras.datasets import mnist
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from time import time
import numpy as np

# load the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"Train shape: {x_train.shape} ")
print(f"Test shape: {x_test.shape} ")

# reshape model to work with (flatten it into 1D)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]**2)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]**2)
print(f"Reshaped train shape: {x_train.shape} ")
print(f"Reshaped test shape: {x_test.shape} ")

# KNN (for K values: 1 to 25)
kValues = np.arange(1, 27, 2)
scores = []
times = []
for i in kValues:
    knn = KNeighborsClassifier(n_neighbors=i)
    startKNNFitTime = time()
    knn.fit(x_train, y_train)
    fitKNNTime = time()-startKNNFitTime
    accuracyKNN = knn.score(x_test, y_test)
    startKNNPredictTime = time()
    knn.predict(x_test)
    predictKNNTime = time()-startKNNPredictTime
    totalKNNTime = fitKNNTime + predictKNNTime
    times.append(totalKNNTime)
    scores.append(accuracyKNN)
    print(f"KNN with {i} neighbor(s) fit time: {fitKNNTime}s \n"
          f"KNN with {i} neighbor(s) predict time: {predictKNNTime}s \n"
          f"KNN with {i} neighbor(s) accuracy: {accuracyKNN} \n")
    print(times)
    print(scores)

# NCC (for distances: Euclidean, Manhattan)
distances = ["manhattan", "euclidean"]
for distance in distances:
    nc = NearestCentroid(distance)
    startNCCFitTime = time()
    nc.fit(x_train, y_train)
    fitNCCTime = time()-startNCCFitTime
    accuracyNCC = nc.score(x_test, y_test)
    startNCCPredictTime = time()
    nc.predict(x_test)
    predictNCCTime = time()-startNCCPredictTime
    totalKNNTime = fitNCCTime + predictNCCTime
    print(f"NCC {distance} fit time: {fitNCCTime}s \n"
          f"NCC {distance} predict time: {predictNCCTime}s \n"
          f"NCC {distance} accuracy: {accuracyNCC} \n")
