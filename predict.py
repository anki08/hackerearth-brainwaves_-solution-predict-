# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:32:03 2016

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

time=test['Time']
print(train.head())
print(train.describe())
print(train.info())
size = 100
col=train.columns
train_corr=train.corr()
threshold=0.90
#finding corr and thn sorting
corr_list=[]
for i in range(0,size): #for 'size' features
    for j in range(i+1,size): #avoid repetition
        if (train_corr.iloc[i,j] >= threshold and train_corr.iloc[i,j] < 1) or (train_corr.iloc[i,j] < 0 and train_corr.iloc[i,j] <= -threshold):
            corr_list.append([train_corr.iloc[i,j],i,j])
s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))

for v,i,j in s_corr_list:
    print ("%s and %s = %.2f" % (col[i],col[j],v))
for v,i,j in s_corr_list:
    sns.pairplot(train, hue="Y", size=6, x_vars=col[i],y_vars=col[j] )
    plt.show()


#train.loc[train["Y"]==-1 , "Y"]=2

size=100
x=col[101]
y=col[1:100]
#violinplot to find distribution of data

for i in range(0,size):
    sns.violinplot(data=train,x=x,y=y[i])  
    plt.show()

#exactlysame distribution
#treating outliners
'''
for i in range(0,size):
    sns.jointplot(data=train,x=x,y=y[i],kind='scatter')  
    plt.show()
'''
y_train=train['Y']
x_train=train.drop(['Time','Y'],axis=1)
#drop 1 of the highly correlated data as both have equal contribution to target
#nooutliners
#we can also use pca to reduce the dimentionality

corr_drop=['X71','X69','X1','X53','X49','X43','X95','X53','X69','X48','X3','X39','X87','X74','X63','X27','X16','X32']
x_train.drop(['X71','X69','X1','X53','X49','X43','X95','X53','X69','X48','X3','X39','X87','X74','X63','X27','X16','X32'],axis=1,inplace=True)
print(x_train.shape)


x_test=test.drop(['Time'],axis=1)
x_test.drop(['X71','X69','X1','X53','X49','X43','X95','X53','X69','X48','X3','X39','X87','X74','X63','X27','X16','X32'],axis=1,inplace=True)

#apply normalizer soall data btw 0 - 1
n = Normalizer()
n.fit(x_train)
x_train = n.transform(x_train)
x_test = n.transform(x_test)
params = {'C':[1],'gamma':[0.001]}
log_reg = SVC()

clf = GridSearchCV(log_reg ,params, refit='True', n_jobs=1, cv=5)


clf.fit(x_train, y_train)

y_test = clf.predict(x_test)
print(clf.score(x_train,y_train))
#print(y_test[:])

#print((clf.score(x_train,y_train)))
print('Best score: {}'.format(clf.best_score_))
print('Best parameters: {}'.format(clf.best_params_))

submission = pd.DataFrame( { 
                  "Y": y_test
                   },index=time)

submission.to_csv('submission3.csv')

