'''
Created on 13 Mar 2019

@author: rampr
'''
from keras.engine.saving import load_model
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import pandas as pd
import numpy as np

Data=pd.read_csv('santander-customer-transaction-prediction/test.csv')
D=Data.drop(['ID_code'],'columns')
X= MinMaxScaler().fit_transform(D.values)

print(X.shape,'\n')



model1=load('model1.joblib')
model2=load('model2.joblib')
model3=load('model3.joblib')
model4=load('model4.joblib')
model5=load('model5.joblib')
model6=load('model6.joblib')
final=load_model('final.model')
print('1')

Xmp=model1.predict(X).reshape(X.shape[0],1)
print('1.1')
Xmp=np.append(Xmp,model2.predict(X).reshape(Xmp.shape[0],1), axis=1)
print('1.2')
Xmp=np.append(Xmp,model3.predict(X).reshape(Xmp.shape[0],1), axis=1)
print('1.3')
Xmp=np.append(Xmp,model4.predict(X).reshape(Xmp.shape[0],1), axis=1)
print('1.4')
Xmp=np.append(Xmp,model5.predict(X).reshape(Xmp.shape[0],1), axis=1)
print('1.5')
Xmp=np.append(Xmp,model6.predict(X).reshape(Xmp.shape[0],1), axis=1)
print('2')

Ypred=final.predict(Xmp)
Ypred=np.around(Ypred)

print(Ypred[:,0])
submission = pd.DataFrame({'ID_code':Data['ID_code'],'target':Ypred[:,0]})
filename = 'Santander_Customer_Transaction_Prediction_1.csv'
submission.to_csv(filename,index=False)

print('Saved file: ' + filename)