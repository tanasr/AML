"""
AML task0
Linear Regression problem
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



#load data sets
data_train = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")
#print(np.mean(train.iloc[0][2:12]))
#X = train.iloc[0][2:12]

#get x and y values for training
x_train = data_train.drop(['Id','y'],axis=1)
y_train = data_train['y']

# create LinearRegression model and fit to data
LR = LinearRegression().fit(x_train,y_train)

# get observations to test the model
x_test = data_test.drop('Id',axis=1)

# predict y values with knowledge from training
y_id = data_test['Id']
y_prediction =  LR.predict(x_test)

#create results which will be submitted 
y_prediction = pd.DataFrame(y_prediction,columns=['y'])
submission = pd.concat([y_id,y_prediction],axis=1)

#create metric to test the model's accuracy
y_test = np.mean(np.transpose(x_test))
RMSE = mean_squared_error(y_test, y_prediction)**0.5
print('the error is: ' + str(RMSE))
# submission.to_csv('out.csv',index=False)


