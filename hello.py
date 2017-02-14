import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

#euclidean distance

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

#Our own classifier K means
import random

class SatanKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return y_train[best_index]
#from sklearn.neighbors import KNeighborsClassifier
my_classifier = SatanKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

#the original test predictions
print(y_test)

#Test accuracy
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))