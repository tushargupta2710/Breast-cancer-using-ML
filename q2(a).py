# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:19:57 2019

@author: Prarthana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 21:55:54 2019

@author: Prarthana
"""
import numpy as np
import matplotlib.pyplot as plt 
def readData(file, col = "\t", row="\n"):
    Lines = open(file, "r").read().split(row)
    return [k.split(col) for k in Lines]

def replaceData(a):
    for i in range(len(a)):
        #workclass
        if(a[i][1]=='Private'):
            a[i][1]=float(-3)
        elif(a[i][1]=='Self-emp-not-inc' or ' Self-emp-not-inc'):
            a[i][1]=float(-2)
        elif(a[i][1]=='Self-emp-inc' or ' Self-emp-inc'):
            a[i][1]=float(-1)
        elif(a[i][1]=='Federal-gov' or ' Federal-gov'):
            a[i][1]=float(0)
        elif(a[i][1]=='Local-gov' or ' Local-gov'):
            a[i][1]=float(1)
        elif(a[i][1]==' State-gov' or 'State-gov'):
            a[i][1]=float(2)
        elif(a[i][1]=='Without-pay' or ' Without-pay'):
            a[i][1]=float(3)
        else:
            a[i][0]=float(4)
        
        if(a[i][11]==0):
            a[i][11]=0.01
        if(a[i][10]==0):
            a[i][10]=0.01
        
        #education
        if(a[i][3]=='Bachelors' or ' Bachelors' ):
            a[i][3]=-8
        elif(a[i][3]=='Some-college' or ' Some-college'):
            a[i][3]=-7
        elif(a[i][3]=='11th' or ' 11th'):
            a[i][3]=-6
        elif(a[i][3]=='HS-grad' or ' HS-grad'):
            a[i][3]=-5
        elif(a[i][3]=='Prof-school' or ' Prof-school'):
            a[i][3]=-4
        elif(a[i][3]=='Assoc-acdm' or ' Assoc-acdm'):
            a[i][3]=-3
        elif(a[i][3]=='Assoc-voc' or ' Assoc-voc'):
            a[i][3]=-2
        elif(a[i][3]=='9th' or ' 9th'):
            a[i][3]=-1
        elif(a[i][3]=='7th-8th' or ' 7th-8th'):
            a[i][3]=0
        elif(a[i][3]=='12th' or ' 12th'):
            a[i][3]=1
        elif(a[i][3]=='Masters' or ' Masters'):
            a[i][3]=2
        elif(a[i][3]=='1st-4th' or ' 1st-4th'):
            a[i][3]=3
        elif(a[i][3]=='10th' or '10th'):
            a[i][3]=4
        elif(a[i][3]=='Doctorate' or 'Doctorate'): 
            a[i][3]=5
        elif(a[i][3]=='5th-6th' or '5th-6th'):
            a[i][3]=6
        elif(a[i][3]=='Preschool' or 'Preschool'):
            a[i][3]=7
        
        #marital-status
        if(a[i][5]=='Married-civ-spouse'):
            a[i][5]=-3
        elif(a[i][5]=='Divorced'):
            a[i][5]=-2
        elif(a[i][5]=='Never-married'):
            a[i][5]=-1
        elif(a[i][5]=='Separated'):
            a[i][5]=0
        elif(a[i][5]=='Widowed'):
            a[i][5]=1
        elif(a[i][5]=='Married-spouse-absent'):
            a[i][5]=2
        elif(a[i][5]=='Married-AF-spouse'):
            a[i][5]=3
        
        #occupation
        if(a[i][6]=='Tech-support'):
            a[i][6]=-7
        elif(a[i][6]=='Craft-repair'):
            a[i][6]=-6
        elif(a[i][6]=='Other-service'):
            a[i][6]=-5
        elif(a[i][6]=='Sales'):
            a[i][6]=-4
        elif(a[i][6]=='Exec-managerial'):
            a[i][6]=-3
        elif(a[i][6]=='Prof-specialty'):
            a[i][6]=-2
        elif(a[i][6]=='Handlers-cleaners'):
            a[i][6]=-1
        elif(a[i][6]=='Machine-op-inspct'):
            a[i][6]=0
        elif(a[i][6]=='Adm-clerical'):
            a[i][6]=1
        elif(a[i][6]=='Farming-fishing'):
            a[i][6]=2
        elif(a[i][6]=='Transport-moving'):
            a[i][6]=3
        elif(a[i][6]=='Priv-house-serv'):
            a[i][6]=4
        elif(a[i][6]=='Protective-serv'):
            a[i][6]=5
        elif(a[i][6]=='Armed-Forces'):
            a[i][6]=6
            
        #relationship
        if(a[i][7]=='Wife'):
            a[i][7]=-2
        elif(a[i][7]=='Own-child'):
            a[i][7]=-1
        elif(a[i][7]=='Husband'):
            a[i][7]=0
        elif(a[i][7]=='Not-in-family'):
            a[i][7]=1
        elif(a[i][7]=='Other-relative'):
            a[i][7]=2
        elif(a[i][7]=='Unmarried'):
            a[i][7]=3
            
        #race
        if(a[i][8]=='White'):
            a[i][8]=-2
        elif(a[i][8]=='Asian-Pac-Islander'):
            a[i][8]=-1
        elif(a[i][8]=='Amer-Indian-Eskimo'):
            a[i][8]=0
        elif(a[i][8]=='Other'):
            a[i][8]=1
        elif(a[i][8]=='Black'):
            a[i][8]=2
        
        #sex
        if(a[i][9]=='Male'):
            a[i][9]=0
        else:
            a[i][9]=1
        
        #native-country
        if(a[i][13]=='United-States'):
            a[i][13]=-20
        elif(a[i][13]=='Cambodia'):
            a[i][13]=-19
        elif(a[i][13]=='England'):
            a[i][13]=-18
        elif(a[i][13]=='Puerto-Rico'):
            a[i][13]=-17
        elif(a[i][13]=='Canada'):
            a[i][13]=-16
        elif(a[i][13]=='Germany'):
            a[i][13]=-15
        elif(a[i][13]=='Outlying-US(Guam-USVI-etc)'):
            a[i][13]=-14
        elif(a[i][13]=='India'):
            a[i][13]=-13
        elif(a[i][13]=='Japan'):
            a[i][13]=-12
        elif(a[i][13]=='Greece'):
            a[i][13]=-11
        elif(a[i][13]=='South'):
            a[i][13]=-10
        elif(a[i][13]=='China'):
            a[i][13]=-9
        elif(a[i][13]=='Cuba'):
            a[i][13]=-8
        elif(a[i][13]=='Iran'):
            a[i][13]=-7
        elif(a[i][13]=='Honduras'):
            a[i][13]=-6
        elif(a[i][13]=='Philippines'):
            a[i][13]=-5
        elif(a[i][13]=='Italy'):
            a[i][13]=-4
        elif(a[i][13]=='Poland'):
            a[i][13]=-3
        elif(a[i][13]=='Jamaica'):
            a[i][13]=-2
        elif(a[i][13]=='Vietnam'):
            a[i][13]=-1
        elif(a[i][13]=='Mexico'):
            a[i][13]=0
        elif(a[i][13]=='Portugal'):
            a[i][13]=1
        elif(a[i][13]=='Ireland'):
            a[i][13]=2
        elif(a[i][13]=='France'):
            a[i][13]=3
        elif(a[i][13]=='Dominican-Republic'):
            a[i][13]=4
        elif(a[i][13]=='Laos'):
            a[i][13]=5
        elif(a[i][13]=='Ecuador'):
            a[i][13]=6
        elif(a[i][13]=='Taiwan'):
            a[i][13]=7
        elif(a[i][13]=='Haiti'):
            a[i][13]=8
        elif(a[i][13]=='Columbia'):
            a[i][13]=9
        elif(a[i][13]=='Hungary'):
            a[i][13]=10
        elif(a[i][13]=='Guatemala'):
            a[i][13]=11
        elif(a[i][13]=='Nicaragua'):
            a[i][13]=12
        elif(a[i][13]=='Scotland'):
            a[i][13]=13
        elif(a[i][13]=='Thailand'):
            a[i][13]=14
        elif(a[i][13]=='Yugoslavia'):
            a[i][13]=15
        elif(a[i][13]=='El-Salvador'):
            a[i][13]=16
        elif(a[i][13]=='Trinadad&Tobago'):
            a[i][13]=17
        elif(a[i][13]=='Peru'):
            a[i][13]=18
        elif(a[i][13]=='Hong'):
            a[i][13]=19
        else:
            a[i][13]=20
        
        #FOR Y
        if(a[i][14]==">50K"):
            a[i][14]=1
        elif(a[i][14]=='<=50K'):
            a[i][14]=0
    return a

def sigmoid(z):
    return 1 / (1 + np.exp(-z))            

def featureNormalise(X):
    X = np.transpose(X)
    for i in range(len(X)):
        if(i==3 or i==1 or i==11 or i==13):
            
            if(X[i].max()!=X[i].min()):
                X[i] = (X[i]-X[i].min())/(X[i].max()-X[i].min())
    return np.transpose(X)


def computeError(h,y):
    c = (h-y)*(h-y)
    l = 0
    for i in range(len(h)):
        l = l +c[i]
    l = (l/len(h))**(1/2)

def gradientDescent(iters, alpha, X, Y, X_test, Y_test, h):
    graphX=[]
    error = []
    errorTest=[]
    for k in range(iters):
        graphX.append(k+1)
        c = np.dot(np.transpose(h),np.transpose(X))
        c = sigmoid(c)
        h= h - (alpha*(np.dot(np.transpose(X),np.transpose(c)))/len(h))
        RMSE=0
        RMSEtest=0
        for i in range(len(X)):
            RMSE += (c[0][i]-Y[i])**2
        for i in range(len(X_test)):
            RMSEtest += (c[0][i]-Y_test[i])**2
        RMSE = RMSE/(2*(len(X)))
        #print(RMSE)
        RMSEtest = RMSEtest/(2*(len(X_test)))
        error.append(RMSE)
        errorTest.append(RMSEtest)
    return graphX,error, errorTest, h
        
def predictOutput(X, theta, mid):
    Expected = np.zeros((30162,1))
    z = sigmoid(np.dot(X, theta))
    for i in range(len(z)):
        if(z[i][0]>=mid):
            Expected[i][0]=0
        else:
            Expected[i][0]=1
    return Expected


a = readData("train.txt")
test = readData("test.txt")
replaceData(test)
replaceData(a)
theta = np.ones((15,1))
b = np.ones((30162,1))
btest = np.ones((len(test),1))
a = np.append(b, a, axis=1)
test = np.append(btest, test, axis=1)
X= []
Y=[]
#print(a)

for i in range(0,len(a)):
    A = []
    for j in range(15):
        A.append(float(a[i][j]))
    X.append(A)
    Y.append(float(a[i][15]))

X_test= []
Y_test=[]


for i in range(0,len(test)):
    A = []
    for j in range(15):
        A.append(float(test[i][j]))
    X_test.append(A)
    Y_test.append(float(test[i][15]))

X= featureNormalise(X)

X_test = featureNormalise(X_test)

    
    
graphX,error, errorTest, theta = gradientDescent(100, 0.00001, X, Y, X_test, Y_test, theta)
fig1 = plt.figure()
plt.xlabel("No. of iterations")
plt.ylabel("Error")
plt.plot(graphX,error, c="blue", label = "Training Set")
plt.legend(loc="upper right")

fig2 = plt.figure()
plt.xlabel("No. of iterations")
plt.ylabel("Error")
plt.plot(graphX,errorTest, c="red", label = "Testing Set")
plt.legend(loc="upper right")

#predicted = sigmoid(np.dot(X, theta))
#predicted = predicted>0.5





