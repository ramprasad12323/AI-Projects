'''
Created on 13 Mar 2019

@author: rampr
'''
from keras.layers import Dense
from xgboost import XGBClassifier
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
import pandas as pd
import numpy as np


Data=pd.read_csv('santander-customer-transaction-prediction/train.csv')
Data=Data.drop(['ID_code'],'columns')

X= MinMaxScaler().fit_transform(Data.values[:,1:])
Y= Data.values[:,0]

print(X.shape,'\n')
print(Y.shape,'\n')
x,X0,y,Y0=train_test_split(X,Y,test_size=0.5,random_state=42)
X1,X2,Y1,Y2=train_test_split(x,y,test_size=0.5,random_state=42)



model1=XGBClassifier()
model2=SGDClassifier(alpha=0.00701,random_state=64,n_iter=4,n_jobs=-1)
model3=AdaBoostClassifier(LogisticRegression(C=9.1,max_iter=1000,verbose=0,n_jobs=-1,solver='sag'))
model4=SVC(coef0=0.9,degree=6,random_state=0,kernel='poly')
model5=KNeighborsClassifier(leaf_size=3,n_neighbors=4,p=2,n_jobs=-1)
model6=RandomForestClassifier(n_estimators=66,max_depth=2,min_samples_leaf=0.1,min_samples_split=0.1,random_state=0,n_jobs=-1)

final = Sequential()
final.add(Dense(units = 1024, input_dim=6, activation = 'relu'))     #Hidden Layer 1 with 512 nods and relu actification function
final.add(Dense(units = 512, activation = 'sigmoid'))     #Hidden Layer 2 with 256 nods and relu actification function
final.add(Dense(units = 256, activation = 'relu'))     #Hidden Layer 2 with 256 nods and relu actification function
final.add(Dense(units = 128, activation = 'sigmoid'))     #Hidden Layer 3 with 5128 nods and relu actification function
final.add(Dense(units = 64, activation = 'sigmoid'))
final.add(Dense(units = 1, activation = 'sigmoid'))   #Output Layer 4 with 62 nods and sigmoid actification function

final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  #Adam Optimizer


model1.fit(X0,Y0)
model2.fit(X0,Y0)
model3.fit(X0,Y0)
model4.fit(X0,Y0)
model5.fit(X0,Y0)
model6.fit(X0,Y0)

Xmp=model1.predict(X1).reshape(X1.shape[0],1)
Xmp=np.append(Xmp,model2.predict(X1).reshape(Xmp.shape[0],1), axis=1)
Xmp=np.append(Xmp,model3.predict(X1).reshape(Xmp.shape[0],1), axis=1)
Xmp=np.append(Xmp,model4.predict(X1).reshape(Xmp.shape[0],1), axis=1)
Xmp=np.append(Xmp,model5.predict(X1).reshape(Xmp.shape[0],1), axis=1)
Xmp=np.append(Xmp,model6.predict(X1).reshape(Xmp.shape[0],1), axis=1)

final.fit(Xmp, Y1, epochs=100, batch_size=500)

dump(model1, 'model1.joblib')
dump(model2, 'model2.joblib') 
dump(model3, 'model3.joblib') 
dump(model4, 'model4.joblib') 
dump(model5, 'model5.joblib') 
dump(model6, 'model6.joblib') 
final.save('final.model')



Xmp1=model1.predict(X2).reshape(X2.shape[0],1)
Xmp1=np.append(Xmp1,model2.predict(X2).reshape(Xmp1.shape[0],1), axis=1)
Xmp1=np.append(Xmp1,model3.predict(X2).reshape(Xmp1.shape[0],1), axis=1)
Xmp1=np.append(Xmp1,model4.predict(X2).reshape(Xmp1.shape[0],1), axis=1)
Xmp1=np.append(Xmp1,model5.predict(X2).reshape(Xmp1.shape[0],1), axis=1)
Xmp1=np.append(Xmp1,model6.predict(X2).reshape(Xmp1.shape[0],1), axis=1)

Ypred=final.predict(Xmp1)
Ypred=np.around(Ypred)

print('Final Roc =',roc_auc_score(Y2,Ypred),'Accuracy =',accuracy_score(Y2,Ypred),'\n')
print('Model0 Roc =',roc_auc_score(Y2,Xmp1[:,0]),'Accuracy =',accuracy_score(Y2,Xmp1[:,0]),'\n')
print('Model1 Roc =',roc_auc_score(Y2,Xmp1[:,1]),'Accuracy =',accuracy_score(Y2,Xmp1[:,1]),'\n')
print('Model2 Roc =',roc_auc_score(Y2,Xmp1[:,2]),'Accuracy =',accuracy_score(Y2,Xmp1[:,2]),'\n')
print('Model3 Roc =',roc_auc_score(Y2,Xmp1[:,3]),'Accuracy =',accuracy_score(Y2,Xmp1[:,3]),'\n')
print('Model4 Roc =',roc_auc_score(Y2,Xmp1[:,4]),'Accuracy =',accuracy_score(Y2,Xmp1[:,4]),'\n')
print('Model5 Roc =',roc_auc_score(Y2,Xmp1[:,5]),'Accuracy =',accuracy_score(Y2,Xmp1[:,5]),'\n')