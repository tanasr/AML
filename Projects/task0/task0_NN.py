"""
AML task0
Linear Regression problem
code from
https://towardsdatascience.com/linear-regression-v-s-neural-networks-cd03b29386d4
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense


#load data sets
data_train = pd.read_csv("train.csv")
# print(data_train.head())
data_test = pd.read_csv("test.csv")
# print(data_train.head())
#print(np.mean(train.iloc[0][2:12]))
#X = train.iloc[0][2:12]

#get x and y values for training
x_train = data_train.drop(columns=['Id','y'],axis=1)
y_train = data_train['y']

#create neural network
network = Sequential()
network.add(Dense(8, input_shape=(10,), activation='relu'))
network.add(Dense(6, activation='relu'))
network.add(Dense(6, activation='relu'))
network.add(Dense(4, activation='relu'))
network.add(Dense(2, activation='relu'))

network.compile('adam', loss='mse', metrics=['mse'])
network.fit(x_train, y_train, epochs=500)

# submission.to_csv('out.csv',index=False)


