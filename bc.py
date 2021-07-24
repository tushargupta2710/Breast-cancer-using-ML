import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import preprocessing

file = pd.read_csv('bc.csv')
#print(file.shape) #569 rows and 33 coulumns 

#print(file.describe)
file = file.drop(columns='id')
file = file.drop(columns='Unnamed: 32')#removing unwanted columns
for i in range(569):
	if (file['diagnosis'][i]=='M'):
		file['diagnosis'][i]=1
	else :
		file['diagnosis'][i]=0


X = file[file.columns[1:31]]
Y = file[file.columns[0]]
#print(X['radius_mean'].max())
#print(X)

X = np.array(X)
#X = featurenorm(X)
#print(X[0][1])

def norm(X):
	X=np.transpose(X)
	print(X)
	for i in range(30):
		for j in range(569):
			X[i][j] = (X[i][j]-X[i].min())/(X[i].max()-X[i].min())
	return np.transpose(X)

X=norm(X)
print(X[0])
#for i in range(569):
	#print(X[i][0])
