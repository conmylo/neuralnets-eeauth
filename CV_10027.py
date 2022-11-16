# import the libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

# load the dataset
digits = load_digits()

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target)

# for 3 neighbors in KNN, for K fold values 2-40 print the mean accuracy
for i in range(2, 40):
    print(sum(cross_val_score(KNeighborsClassifier(n_neighbors=3), digits.data, digits.target, cv=i))/i)

