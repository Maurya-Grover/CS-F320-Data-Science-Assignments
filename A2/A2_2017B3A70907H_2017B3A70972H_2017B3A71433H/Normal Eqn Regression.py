# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from random import seed
import numpy as np
import pandas as pd
import math

#Split dataset into training and testing set
def train_test_split(dataframe,split=0.70):
    train_size = int(split * len(dataframe))    #print(train_size)
    test_size = len(dataframe) - train_size     #print(test_size)
    dataframe = dataframe.sample(frac=1)        #shuffle rows of the dataframe     
    train = dataframe[:train_size]              #copy first 70% elements into train
    test = dataframe[-test_size:]               #copy last 30% elements into test
    return train,test

def RMSE(df,wt):
    N = len(df)
    X = df.loc[:,'bias':'children']
    y_obs = df.loc[:,'charges']
    
    yi_hat = X.dot(wt)
    diff = (y_obs - yi_hat)**2
    rmse = math.sqrt(np.sum(diff)/N)
    return rmse

    
def Linear_Regression(dataframe):
    train, test = train_test_split(dataframe)   
    
    X = train.loc[:,'bias':'children']
    Y = train.loc[:,'charges']
    
    # Formula to obtain weight matrix :  w=((XT*X)−1)*XT*y , where XT = X_transpose
    X_transpose = np.transpose(X)
    mul = np.dot(X_transpose,X)
    Inv_matrix = np.linalg.inv(mul)
    mul2 = np.dot(Inv_matrix,X_transpose)
    result = np.dot(mul2,Y)
    
    test_rmse = RMSE(test,result)
    train_rmse = RMSE(train, result)
    return result,train_rmse,test_rmse
    

if __name__ == '__main__':
    seed(1)
    dataframe = pd.read_csv('D:/4-1/FODS/insurance.txt', sep=",",header=None)
    dataframe.columns = ["age","bmi","children","charges"]
    dataframe = pd.concat([pd.Series(1,index=dataframe.index,name="bias"),dataframe],axis=1)
    
    #Normalize the input variables by dividing each column by the maximum values of that column
    for column in dataframe:
        max_val = np.max(dataframe[column])
        dataframe[column]=dataframe[column]/max_val
    #print(dataframe[:5])
    
    #initializing variabless
    num_samples = 20
    rows,cols = (num_samples,4)
    weights = [[0]*cols]*rows
    train_data_error = np.zeros(num_samples)
    test_data_error = np.zeros(num_samples)
    
    w0_sum,w1_sum,w2_sum,w3_sum = (0,0,0,0)
    err_train_sum,err_test_sum = (0,0)
    
    #Sampling data 20 times to get more accurate results
    for i in range(0,num_samples):
        weights[i],train_data_error[i],test_data_error[i] = Linear_Regression(dataframe)
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
    
    print("\nRMSE mean of prediction of training data: ",train_mean_err)
    print("RMSE variance of prediction of training data: ",var_train_error)
    #print("Minimum RMSE of training data: ",min_train_error)
    
    print("\nRMSE Mean of prediction of testing data: ",test_mean_err)
    print("RMSE variance of prediction of testing data: ",var_test_error)
    #print("Minimum RMSE of testing data: ",min_test_error)
    print("\nMinimum RMSE of regression model using normal equation",min(min_train_error,min_test_error))