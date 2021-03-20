# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 06:31:12 2020

@author: Siddhi
"""

from random import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Split dataset into training and testing set
def train_test_split(dataframe,split=0.70):
    train_size = int(split * len(dataframe))    #print(train_size)
    test_size = len(dataframe) - train_size     #print(test_size)
    dataframe = dataframe.sample(frac=1)        #shuffle rows of the dataframe     
    train = dataframe[:train_size]              #copy first 70% elements into train
    test = dataframe[-test_size:]               #copy last 30% elements into test
    return train,test

def hypothesis(weight,X):
    return weight*X


def Compute_Cost(X,y,weight):
    m = X.shape[0]
    y1 = hypothesis(weight,X)
    y1 = np.sum(y1,axis=1)
    return sum((y1-y)**2)/(2*m)


def SGD(X,Y,weights,alpha,epoch,precision):
    m = X.shape[0]
    cost_history = np.zeros(epoch)
    
    for i in range(epoch):
        cost = 0.0
        df = pd.concat([X,Y],axis=1)
        df = df.sample(frac=1)
        X = df.loc[:,'bias':'children']
        Y = df.loc[:,'charges']
        for j in range(0,m):
            X_j = X.iloc[j] 
            y_prediction = X_j.dot(weights)
            X_T = X_j.transpose()
            diff = y_prediction - Y.iloc[j]
            weights = weights - alpha*(X_T * diff)
        
        cost += Compute_Cost(X,Y,weights)
        if(cost<=precision):
            break
        cost_history[i]=cost
        
    plt.plot(cost_history)
    plt.title("learning rate = 0.005")
    plt.xlabel('number of epoch')
    plt.ylabel('loss')
    return cost_history,weights


def RMSE(df,wt):
    N = len(df)
    X = df.loc[:,'bias':'children']
    y_obs = df.loc[:,'charges']
    
    yi_hat = X.dot(wt)
    diff = (y_obs - yi_hat)**2
    rmse = math.sqrt(np.sum(diff)/N)
    return rmse


def Solve(dataframe):
    #Splitting the data into training and testing data
    train, test = train_test_split(dataframe)
    #print(train.shape)
    X = train.loc[:,'bias':'children']
    Y = train.loc[:,'charges']
    
    weight = np.array([0.0]*len(X.columns))
    learning_rate = 0.005  #0.005
    epoch = 151
    precision = 0.000001
    
    Cost,weight = SGD(X,Y,weight,learning_rate,epoch,precision)
    print("Cost Values after every 50 epochs: ")
    for i in range(0,len(Cost)):
        if(i%50==0):
            print(Cost[i])
    
    train_loss = RMSE(train,weight)
    test_loss = RMSE(test,weight)
    return weight,train_loss, test_loss
    
            
if __name__ == '__main__':
    seed(1)
    dataframe = pd.read_csv('D:/Study/4-1/Fundamentals of Data Science/Assignments/A2/A2_2017B3A70907H_2017B3A70972H_2017B3A71433H/insurance.txt', sep=",",header=None)
    dataframe.columns = ["age","bmi","children","charges"]
    dataframe = pd.concat([pd.Series(1,index=dataframe.index,name="bias"),dataframe],axis=1)
    #Normalize the input variables by dividing each column by the maximum values of that column
    for column in dataframe:
        max_val = np.max(dataframe[column])
        dataframe[column]=dataframe[column]/max_val
    #print(dataframe[:5])

    #initializing the variables
    num_samples = 20
    rows,cols = (num_samples,3)
    weights = np.zeros((num_samples,4))
    b_val = np.zeros(num_samples)
    train_data_error = np.zeros(num_samples)
    test_data_error = np.zeros(num_samples)
    
    w0_sum,w1_sum,w2_sum,w3_sum = (0,0,0,0)
    err_train_sum,err_test_sum = (0,0)
    
    #Sampling the data 20 times to get more accurate results
    for i in range(0,num_samples):
        print("\nSample ",i+1)
        weights[i],train_data_error[i],test_data_error[i] = Solve(dataframe)
        w0_sum += weights[i][0]
        w1_sum += weights[i][1]
        w2_sum += weights[i][2]
        w3_sum += weights[i][3]
        err_train_sum += train_data_error[i]
        err_test_sum += test_data_error[i]
        
    train_mean_err = err_train_sum/num_samples
    test_mean_err = err_test_sum/num_samples
    var_train_error = np.var(train_data_error)
    var_test_error = np.var(test_data_error)
    min_train_error = np.amin(train_data_error)
    min_test_error = np.amin(test_data_error)

    print("\nThe regression line is of the form:\n insurance = w0 + w1*age + w2*bmi + w3*children")    
    result = [w0_sum/num_samples, w1_sum/num_samples, w2_sum/num_samples, w3_sum/num_samples]
    print("\nWeights of the independent variables are: ")
    print("w0 = ",result[0])
    print("w1 = ",result[1])
    print("w2 = ",result[2])
    print("w3 = ",result[3])
    
    print("\nRMSE Mean of accuracy prediction of training data: ",train_mean_err)
    print("RMSE Variance of accuracy prediction of training data: ",var_train_error)
    #print("Minimum RMSE of training data: ",min_train_error)
    
    print("\nRMSE Mean of accuracy prediction of testing data: ",test_mean_err)
    print("RMSE Variance of accuracy prediction of testing data: ",var_test_error)
    #print("Minimum RMSE of testing data: ",min_test_error)
    print("\nMinimum RMSE of regression model using normal equation",min(min_train_error,min_test_error))
    
