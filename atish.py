import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from sklearn import  svm

# References
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html

# Reading dataset
data_pd = pd.read_csv('bc.csv')
data = data_pd.values
size = 569

# storing Diagnosis in result
result = [data[x][1] for x in range(size)]
for x in range(569):
    if result[x] == 'M':
        result[x] = 0
    else:
        result[x] = 1

# Dropping diagnosis and id
data_pd = data_pd.drop(columns='diagnosis')
data_pd = data_pd.drop(columns='id')
data = data_pd.values
final_set = [[data[j][i] for i in range(30)] for j in range(size)]

# Selecting K best features out of 30 features
set_new = SelectKBest(chi2, k=10).fit_transform(final_set, result)

# setting final variables
X = set_new
Y = result
print(X)


# function for normalization
def norm(x):
    x = np.transpose(x)
    for i in range(len(x)):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    return np.transpose(x)


# Doing normalization on X
X = norm(X)

# Train-test split ( Using 80% for training and 20% for testing)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, Y_train)

Y_predicted = knn.predict(X_test)
a = 0
b = 0
for i in range(len(Y_predicted)):
    if Y_predicted[i] == Y_test[i]:
        a = a+1
    else:
        b = b+1
print(a/(a+b))
print(b)


classifier = SVC(kernel = 'rbf', random_state = 1)

trained_model=classifier.fit(X_train,Y_train)
trained_model.fit(X_train,Y_train )


# Predicting the Test set results

Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

CM= confusion_matrix(Y_test, Y_pred)
print(CM)
a1= accuracy_score(Y_train, trained_model.predict(X_train))*100
print("Accuracy of train SVM")
print(a1)
print("Accuracy of test SVM")
a2= accuracy_score(Y_test, Y_pred)*100
print(a2)

para1 = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
para2 = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
def bestparam_for_poly(xtrian,ytrain, nopffolds):
	pa = {'C':para1 , 'degree':para2}
	best = GridSearchCV(svm.SVC(kernel='poly'),pa,cv=nopffolds)
	best.fit(xtrian,ytrain).best_params_
	return best_params_

print(bestparam_for_poly(X_train,Y_train,10))





























