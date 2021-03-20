# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 21:03:52 2020

@author: Siddhi and Maurya
"""

from random import seed
import numpy as np
import pandas as pd
import math
# import matplotlib.pyplot as plt


# Split dataset into training and testing set
def train_test_split(dataframe, split=0.70):
    train_size = int(split * len(dataframe))    # print(train_size)
    test_size = len(dataframe) - train_size     # print(test_size)
    dataframe = dataframe.sample(frac=1)        # shuffle rows of the dataframe
    train = dataframe[:train_size]              # copy first 70% elements into train
    test = dataframe[-test_size:]               # copy last 30% elements into test
    return train, test


def hypothesis(weight, X):
    return weight*X


def Compute_Cost(X, y, weight):
    m = X.shape[0]
    y1 = hypothesis(weight, X)
    y1 = np.sum(y1, axis=1)
    return sum((y1-y)**2)/(2*m)


def RMSE(df, wt):
    N = len(df)
    X = df.loc[:, 'bias':'children']
    y_obs = df.loc[:, 'charges']

    yi_hat = X.dot(wt)
    diff = (y_obs - yi_hat)**2
    rmse = math.sqrt(np.sum(diff)/N)
    return rmse


def Solving_normal_eqn(X, Y):
    # Formula to obtain weight matrix :  w=((XT*X)−1)*XT*y , where XT = X_transpose
    X_transpose = np.transpose(X)
    mul = np.dot(X_transpose, X)
    Inv_matrix = np.linalg.inv(mul)
    mul2 = np.dot(Inv_matrix, X_transpose)
    result = np.dot(mul2, Y)

    return result


def Gradient_Descent(X, y, weight, alpha, epoch, precision):
    N = len(X)
    Cost_fn = []
    i = 0
    while(i < epoch):
        y_predicted = hypothesis(weight, X)
        y_predicted = np.sum(y_predicted, axis=1)

        for c in range(0, len(X.columns)):
            weight[c] = weight[c] - alpha*(sum((y_predicted-y)*X.iloc[:, c])/N)

        j = Compute_Cost(X, y, weight)
        Cost_fn.append(j)
        if(i % 100 == 0):
            print(j)
        if(j < precision):
            break
        i += 1

    return weight


def SGD(X, Y, weights, alpha, epoch, precision):
    m = X.shape[0]
    cost_history = np.zeros(epoch)

    for i in range(epoch):
        cost = 0.0
        df = pd.concat([X, Y], axis=1)
        df = df.sample(frac=1)
        X = df.loc[:, 'bias':'children']
        Y = df.loc[:, 'charges']
        for j in range(0, m):
            X_j = X.iloc[j]
            y_prediction = X_j.dot(weights)
            X_T = X_j.transpose()
            diff = y_prediction - Y.iloc[j]
            weights = weights - alpha*(X_T * diff)

        cost += Compute_Cost(X, Y, weights)
        if(i % 50 == 0):
            print(cost)
        if(cost <= precision):
            break
        cost_history[i] = cost

    return weights


def Run_regr(dataframe):
    train, test = train_test_split(dataframe)
    X = train.loc[:, 'bias':'children']  # selecting all rows and columns from bias to children
    Y = train.loc[:, 'charges']
    # print(X)
    # print(Y)

    # build regression model by solving normal eqn
    NE_wt = Solving_normal_eqn(X, Y)
    print("Solving using Normal eq weights: ", NE_wt)
    NE_train_err = RMSE(train, NE_wt)
    NE_test_err = RMSE(test, NE_wt)

    # build regression model using GD
    weight = np.array([0.0]*len(X.columns))         # initializing parameters
    GD_Lr = 0.9
    precision = 0.00001
    GD_epoch = 500
    print("\nGradient Descent: ")
    print("Cost Values after every 100 epochs: ")
    GD_wt = Gradient_Descent(X, Y, weight, GD_Lr, GD_epoch, precision)
    print("GD weights: ", GD_wt)
    GD_train_err = RMSE(train, GD_wt)
    GD_test_err = RMSE(test, GD_wt)

    # build regression model using SGD
    SGD_epoch = 150
    SGD_Lr = 0.005
    print("\nStochastic Gradient Descent: ")
    print("Cost values after every 50 epochs: ")
    SGD_wt = SGD(X, Y, weight, SGD_Lr, SGD_epoch, precision)
    print("SGD weights: ", SGD_wt[0], SGD_wt[1], SGD_wt[2], SGD_wt[3])
    SGD_train_err = RMSE(train, SGD_wt)
    SGD_test_err = RMSE(test, SGD_wt)

    return NE_wt, NE_train_err, NE_test_err, GD_wt, GD_train_err, GD_test_err, SGD_wt, SGD_train_err, SGD_test_err


def Prediction_accuracy(NE_weights, NE_train_error, NE_test_error, GD_weights, GD_train_error, GD_test_error, SGD_weights, SGD_train_error, SGD_test_error):
    NE_train_err_mean = np.mean(NE_train_error)
    NE_train_err_var = np.var(NE_train_error)
    NE_test_err_mean = np.mean(NE_test_error)
    NE_test_err_var = np.var(NE_test_error)
    NE_test_err_min = np.amin(NE_test_error)

    print("\nRegression solving Normal Eqns accuracy:")
    print("Weights of regression: ", np.mean(NE_weights, axis=0))
    print("RMSE mean of prediction of training data: ", NE_train_err_mean)
    print("RMSE variance of prediction of training data: ", NE_train_err_var)
    print("RMSE Mean of prediction of testing data: ", NE_test_err_mean)
    print("RMSE variance of prediction of testing data: ", NE_test_err_var)
    print("Minimum RMSE of regression model using normal equation", NE_test_err_min)

    GD_train_err_mean = np.mean(GD_train_error)
    GD_train_err_var = np.var(GD_train_error)
    GD_test_err_mean = np.mean(GD_test_error)
    GD_test_err_var = np.var(GD_test_error)
    GD_test_err_min = np.amin(GD_test_error)
    print("\nRegression using Gradient Descent accuracy:")
    print("Weights of regression: ", np.mean(GD_weights, axis=0))
    print("RMSE mean of prediction of training data: ", GD_train_err_mean)
    print("RMSE variance of prediction of training data: ", GD_train_err_var)
    print("RMSE Mean of prediction of testing data: ", GD_test_err_mean)
    print("RMSE variance of prediction of testing data: ", GD_test_err_var)
    print("Minimum RMSE of regression model using normal equation", GD_test_err_min)

    SGD_train_err_mean = np.mean(SGD_train_error)
    SGD_train_err_var = np.var(SGD_train_error)
    SGD_test_err_mean = np.mean(SGD_test_error)
    SGD_test_err_var = np.var(SGD_test_error)
    SGD_test_err_min = np.amin(SGD_test_error)
    print("\nRegression using Stochastic Gradient Descent accuracy:")
    print("Weights of regression: ", np.mean(SGD_weights, axis=0))
    print("RMSE mean of prediction of training data: ", SGD_train_err_mean)
    print("RMSE variance of prediction of training data: ", SGD_train_err_var)
    print("RMSE Mean of prediction of testing data: ", SGD_test_err_mean)
    print("RMSE variance of prediction of testing data: ", SGD_test_err_var)
    print("Minimum RMSE of regression model using normal equation", SGD_test_err_min)

    return


if __name__ == '__main__':
    seed(1)
    dataframe = pd.read_csv('D:/Study/4-1/Fundamentals of Data Science/Assignments/A2/A2_2017B3A70907H_2017B3A70972H_2017B3A71433H/insurance.txt', sep=",", header=None)
    dataframe.columns = ["age", "bmi", "children", "charges"]
    dataframe = pd.concat([pd.Series(1, index=dataframe.index, name="bias"), dataframe], axis=1)
    # print(dataframe)
    # Normalize the input variables by dividing each column by the maximum values of that column
    for column in dataframe:
        max_val = np.max(dataframe[column])
        dataframe[column] = dataframe[column]/max_val
    print(dataframe[:5])

    num_samples = 3
    rows, cols = (num_samples, 4)

    NE_weights = [[0]*cols]*rows                # initializing arrays to store values of 20 NE models
    NE_train_error = np.zeros(num_samples)
    NE_test_error = np.zeros(num_samples)

    GD_weights = [[0]*cols]*rows                # initializing arrays to store values of 20 GD models
    GD_train_error = np.zeros(num_samples)
    GD_test_error = np.zeros(num_samples)

    SGD_weights = [[0]*cols]*rows                # initializing arrays to store values of 20 SGD models
    SGD_train_error = np.zeros(num_samples)
    SGD_test_error = np.zeros(num_samples)

    for i in range(0, num_samples):
        print("\nSample ", i+1)
        NE_weights[i], NE_train_error[i], NE_test_error[i], GD_weights[i], GD_train_error[i], GD_test_error[i], SGD_weights[i], SGD_train_error[i], SGD_test_error[i] = Run_regr(dataframe)

    Prediction_accuracy(NE_weights, NE_train_error, NE_test_error, GD_weights, GD_train_error, GD_test_error, SGD_weights, SGD_train_error, SGD_test_error)