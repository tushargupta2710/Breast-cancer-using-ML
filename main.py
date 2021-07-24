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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import  svm
import warnings as w

w.simplefilter(action='ignore', category=FutureWarning)

# References
# https://scikit-learn.org/stable/modules/feature_selection.html
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
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

# setting final variables
X = final_set
Y = result
Y = np.array(result)
# print(X)


# function for normalization
def norm(x):
    x = np.transpose(x)
    for i in range(len(x)):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    return np.transpose(x)


# Doing normalization on X
X = norm(X)


# Selecting K best features out of 30 features
set_new = SelectKBest(chi2, k=15).fit(X, Y)
dfscores = pd.DataFrame(set_new.scores_)
dfcolumns = pd.DataFrame(data_pd.columns)

# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  #naming the dataframe columns
feature = featureScores.nlargest(15, 'Score')



# Train-test split ( Using 80% for training and 20% for testing)
#variables for mean accuracy
meana1=0
meana2=0
lista1=[]
lista2=[]
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(X, Y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    para1 = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
    para2 = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
    def bestparam_for(xtrian,ytrain, nopffolds):
        parameters = [{'kernel': ['rbf'], 'gamma': para1,'C': para2}]
        pa = {'C':para1 , 'degree':para2}
        best = GridSearchCV(SVC(),parameters,cv=nopffolds)
        best.fit(xtrian,ytrain)
        return best.best_params_

    params = bestparam_for(X_train,Y_train,10)
    print(bestparam_for(X_train,Y_train,10))


    classifier = SVC(kernel = 'rbf',C=10,gamma = 0.1, random_state = 1)
    trained_model=classifier.fit(X_train,Y_train)
    trained_model.fit(X_train,Y_train )


# Predicting the Test set results

    Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

    CM= confusion_matrix(Y_test, Y_pred)
    print(CM)

    a1= accuracy_score(Y_train, trained_model.predict(X_train))*100
    a2= accuracy_score(Y_test, Y_pred)*100
    meana2 = meana2+a2
    meana1 = meana1+a1
    lista1.append(a1)
    lista2.append(a2)


xaxis = [1,2,3,4,5]

print("average Accuracy of train SVM")    
print(meana1/5)
print("average Accuracy of test SVM")
print(meana2/5)

#grpah accuracy vs fold

plt.plot(xaxis, lista1, color='green', linestyle='dashed', linewidth = 3,marker='o', markerfacecolor='blue', markersize=12)  
plt.ylim(90,100) 
plt.xlim(0,6) 
plt.xlabel('set number')
plt.ylabel('accuracy of train svm')
plt.show()

plt.plot(xaxis, lista2, color='yellow', linestyle='dashed', linewidth = 3, marker='o', markerfacecolor='red', markersize=12) 
plt.ylim(90,100) 
plt.xlim(0,6) 
plt.xlabel('set number')
plt.ylabel('accuracy of test svm')
plt.show()































