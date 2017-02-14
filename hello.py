import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


#decision tree

from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

#the original test predictions
print(y_test)

#Test accuracy
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)
print(predictions)

#the original test predictions
print(y_test)

#Test accuracy
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, predictions))