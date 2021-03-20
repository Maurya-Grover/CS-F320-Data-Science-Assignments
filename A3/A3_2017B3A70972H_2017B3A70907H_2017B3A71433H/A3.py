# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:40:02 2020

@author: Siddhi, Parth, Maurya
"""
from random import seed
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def data_normalize(df):
    for column in df:
        max_val = np.max(df[column])
        df[column]=df[column]/max_val
    #print(df[:5]) 
    return df
    
def split_df(df):
    df = df.sample(frac=1)
    train_size = int(0.70 * len(df))
    validation_size = int(0.20 * len(df))
    test_size = len(df) - train_size - validation_size
    
    train_data = df[:train_size]
    validation_data = df[train_size:train_size+validation_size]
    test_data = df[-test_size:]
    
    return train_data, validation_data, test_data

def PreProcess(df,degree_i):
    data = df.to_numpy()
    X,Y = data[:,:-1], data[:,-1]
    poly = PolynomialFeatures(degree=degree_i)
    X_poly = poly.fit_transform(X)
    X_df = pd.DataFrame(data = X_poly) 
    Y_df = pd.DataFrame(data = Y)
    df = pd.concat([X_df,Y_df],axis=1)
    df = data_normalize(df)

    return df

def predicted(X,theta):
    y_pred = np.dot(theta,X.T)
    y_pred = y_pred.reshape(-1,1)
    return y_pred

def Compute_Cost(X,y,weight):
    m = X.shape[0]
    y1 = predicted(X,weight)
    cost = np.sum((y1-y)**2)/(2*m)
    return cost
 
def Lasso_cost(X,y,theta,lambda_):
    return  (((1/2)*np.sum(np.square((predicted(X,theta))-y)))+(lambda_*(np.sum(np.absolute(theta)))))/len(X)

def Ridge_cost(X,y,theta,lambda_):
    return  (((1/2)*np.sum(np.square((predicted(X,theta))-y)))+(lambda_*(np.sum(np.square(theta)))))/len(X)

def rmse(df,final_weights):
    X = df.iloc[:,0:len(df.columns)-1].to_numpy()
    y = df.iloc[:,-1:].to_numpy()
    N = len(df)
    yi_hat = X.dot(final_weights.T)
    diff = np.square(y - yi_hat)
    rmse = math.sqrt(np.sum(diff)/N)
    return rmse


def Gradient_descent(X,Y,weights,lr,epochs,precision):
    N = len(X)
    weights = weights.reshape(-1,1)
    weights = weights.T
    Cost = np.zeros(epochs)
    
    for i in range(epochs):
        y_pred = predicted(X,weights)
        weights = weights - (lr/N) * np.dot((y_pred-Y).T,X)
        Cost[i] = Compute_Cost(X,Y,weights)
        if((i>=1) & (Cost[i-1]-Cost[i]<=precision)):
            break
    
    """for i in range(epochs):
        if(i%20==0):
            print(Cost[i])"""
    #print(weights)
    return weights

def GD_Lasso_reg(X,Y,weights,lr,epochs,lambda_,precision):
    N = len(X)
    weights = weights.reshape(-1,1)
    weights = weights.T
    Cost = np.zeros(epochs)
     
    for i in range(epochs):
       y_pred = predicted(X,weights)
       weights = weights - (lr/N) * (np.dot((y_pred-Y).T,X) + lambda_*np.sign(weights))
       Cost[i] = Lasso_cost(X,Y,weights,lambda_)
       if((i>=1) & (Cost[i-1]-Cost[i]<=precision)):
            break
        
    """for i in range(epochs):
       if(i%20==0):
           print(Cost[i])"""
    
    #print(weights)
    return weights

def GD_Ridge_reg(X,Y,weights,lr,epochs,lambda_,precision):
    N = len(X)
    weights = weights.reshape(-1,1)
    weights = weights.T
    Cost = np.zeros(epochs)
    
    for i in range(epochs):
        y_pred = predicted(X,weights)
        weights = weights - (lr/N) * (np.dot((y_pred-Y).T,X) + lambda_*weights)
        Cost[i] = Ridge_cost(X,Y,weights,lambda_)
        if((i>=1) & (Cost[i-1]-Cost[i]<=precision)):
            break
    """for i in range(epochs):
        if(i%20==0):
            print(Cost[i])"""
    
    #print(weights)
    return weights

def SGD(df,weights,lr,epochs,precision):
    N = len(df)
    weights = weights.reshape(-1,1)
    weights = weights.T
    Cost = np.zeros(epochs)
    
    for i in range(epochs):
        cost = 0.0
        df = df.sample(frac=1)
        X = df.iloc[:,0:len(df.columns)-1]
        Y = df.iloc[:,-1:]
        X = X.to_numpy()
        Y = Y.to_numpy()
        for j in range(0,N):
            X_j = (X[j].reshape(-1,1)).T
            y_pred = X_j.dot(weights.T)
            y_obs = Y[j].reshape(-1,1)
            weights = weights - (lr) * ((y_pred-y_obs).T * X_j)
        cost += Compute_Cost(X,Y,weights)
        Cost[i] = cost
        if((i>=1) & (Cost[i-1]-Cost[i]<=precision)):
            break
            
    #print(weights)
    return weights

def SGD_Lasso_reg(df,weights,lr,epochs,lambda_,precision):
    N = len(df)
    weights = weights.reshape(-1,1)
    weights = weights.T
    Cost = np.zeros(epochs)
    
    for i in range(epochs):
        cost = 0.0
        df = df.sample(frac=1)
        X = df.iloc[:,0:len(df.columns)-1]
        Y = df.iloc[:,-1:]
        X = X.to_numpy()
        Y = Y.to_numpy()
        for j in range(0,N):
            X_j = (X[j].reshape(-1,1)).T
            y_pred = X_j.dot(weights.T)
            y_obs = Y[j].reshape(-1,1)
            weights = weights - (lr) * (((y_pred-y_obs).T * X_j) + lambda_*np.sign(weights))
        cost += Compute_Cost(X,Y,weights)
        Cost[i] = cost
        if((i>=1) & (Cost[i-1]-Cost[i]<=precision)):
            break
        
    #print(weights)
    return weights

def SGD_Ridge_reg(df,weights,lr,epochs,lambda_,precision):
    N = len(df)
    weights = weights.reshape(-1,1)
    weights = weights.T
    Cost = np.zeros(epochs)
    
    for i in range(epochs):
        cost = 0.0
        df = df.sample(frac=1)
        X = df.iloc[:,0:len(df.columns)-1]
        Y = df.iloc[:,-1:]
        X = X.to_numpy()
        Y = Y.to_numpy()
        for j in range(0,N):
            X_j = (X[j].reshape(-1,1)).T
            y_pred = X_j.dot(weights.T)
            y_obs = Y[j].reshape(-1,1)
            weights = weights - (lr) * (((y_pred-y_obs).T * X_j) + lambda_*weights)
        cost += Compute_Cost(X,Y,weights)
        Cost[i] = cost
        if((i>=1) & (Cost[i-1]-Cost[i]<=precision)):
            break
            
    #print(weights)
    return weights


def Plot_surface(train,x1,x2,deg,weights):
    X_train = train.iloc[:,0:len(train.columns)-1]
    Y_train = train.iloc[:,-1:]
    X_tr = X_train.to_numpy()
    Y_tr = Y_train.to_numpy()
    
    z = X_tr.dot(weights.T)
    
    fig = plt.figure()
    my_map = plt.get_cmap('hot')
    ax = fig.add_subplot(111,projection='3d')
    ax.set_title('POLYNOMIAL OF DEGREE {}'.format(deg),fontsize=10,color='Blue')
    ax.set_xlabel('BMI',fontsize=10,color='red',y=5)
    ax.set_ylabel('AGE',color='Green',fontsize=10,y=5)
    ax.set_zlabel('INSURANCE',color='Purple',fontsize=10,y=5)
    trisurf = ax.plot_trisurf(x1.flatten(),x2.flatten(),z.flatten(),cmap=my_map,linewidth=0.2,antialiased=True,edgecolor='grey')
    ax.scatter(x1[:70].flatten(), x2[:70].flatten(), Y_tr[:70].flatten(), zdir='z', s=20, c=None, depthshade=True, cmap=my_map)
    plt.show()
    

if __name__ == '__main__':
    seed(1)
    dataframe = pd.read_csv('D:/4-1/FODS/A3/insurance.txt', sep=",",header=None)
    #dataframe = pd.read_csv('/home/siddhi/Documents/4-1/FODS/A3/insurance.txt', sep=",",header=None)
    dataframe.columns = ["age","bmi","children","charges"]
    dataframe = dataframe.drop(['children'],axis=1)
    
    train_data, validation_data, test_data = split_df(dataframe)
    df_temp = data_normalize(train_data)
    print(df_temp[:5])
    x1 = df_temp['age'].to_numpy()
    x2 = df_temp['bmi'].to_numpy()
    
    """lambda_rand = np.zeros(10)
    for i in range(0,10):
        lambda_rand[i] = random.uniform(0,1)
    lambda_rand = np.sort(lcambda_rand)
    print(lambda_rand)"""
    lambda_rand = [0.02834748, 0.09385959, 0.13436424, 0.25506903, 0.44949106, 0.49543509, 0.65159297, 0.76377462, 0.78872335, 0.84743374]

    for i in range(1,11):
        print("\ndegree",i)
        if(i==1):
            train_data_1 = pd.concat([pd.Series(1,index=train_data.index,name="bias"),train_data],axis=1)
            validation_data_1 = pd.concat([pd.Series(1,index=validation_data.index,name="bias"),validation_data],axis=1)
            test_data_1 = pd.concat([pd.Series(1,index=test_data.index,name="bias"),test_data],axis=1)
            train,validation,test = data_normalize(train_data_1),data_normalize(validation_data_1),data_normalize(test_data_1)
        else:
            train,validation,test = PreProcess(train_data,i), PreProcess(validation_data,i), PreProcess(test_data,i)
        
        X = train.iloc[:,0:len(train.columns)-1]
        Y = train.iloc[:,-1:]
        weight = np.zeros(len(train.columns)-1)
        
        """GD_lr, GD_epochs = 0.01,1000
        SGD_lr, SGD_epochs = 0.0005, 100
        precision = 0.000001"""
        
        result_gd_L1 = np.zeros((10,4))
        result_gd_L2 = np.zeros((10,4))
        result_sgd_L1 = np.zeros((10,4))
        result_sgd_L2 = np.zeros((10,4))
        result_gd_L1[:,0] = lambda_rand
        result_gd_L2[:,0] = lambda_rand
        result_sgd_L1[:,0] = lambda_rand
        result_sgd_L2[:,0] = lambda_rand
           
        gd_L1_column_title = ['Reg Const','GD_L1_train','GD_L1_validate','GD_L1_test']
        gd_L2_column_title = ['Reg Const','GD_L2_train','GD_L2_validate','GD_L2_test']
        sgd_L1_column_title = ['Reg Const','SGD_L1_train','SGD_L1_validate','SGD_L1_test']
        sgd_L2_column_title = ['Reg Const','SGD_L2_train','SGD_L2_validate','SGD_L2_test']
        
        count =0
        for j in lambda_rand:
            lambda_ = j
            precision = 0.000001
            GD_lr, GD_epochs = 0.01, 1000
            #print("lambda ",lambda_)
            #print("GD")
            GD_wt = Gradient_descent(X,Y,weight,GD_lr,GD_epochs,precision)
            GD_train_rmse ,GD_valid_rmse, GD_test_rmse = rmse(train,GD_wt),rmse(validation,GD_wt), rmse(test,GD_wt)
            #print(GD_train_rmse,GD_valid_rmse,GD_test_rmse)
            
            GD_L1_wt = GD_Lasso_reg(X,Y,weight,GD_lr,GD_epochs,lambda_,precision)
            GD_train_rmse ,GD_valid_rmse, GD_test_rmse = rmse(train,GD_L1_wt),rmse(validation,GD_L1_wt), rmse(test,GD_L1_wt)
            result_gd_L1[count][1], result_gd_L1[count][2], result_gd_L1[count][3] = GD_train_rmse ,GD_valid_rmse, GD_test_rmse
            #print(GD_train_rmse,GD_valid_rmse,GD_test_rmse)
            
            
            GD_L2_wt = GD_Ridge_reg(X,Y,weight,GD_lr,GD_epochs,lambda_,precision)
            GD_train_rmse , GD_valid_rmse,GD_test_rmse = rmse(train,GD_L2_wt),rmse(validation,GD_L2_wt), rmse(test,GD_L2_wt)
            result_gd_L2[count][1], result_gd_L2[count][2], result_gd_L2[count][3] = GD_train_rmse , GD_valid_rmse,GD_test_rmse 
            #print(GD_train_rmse,GD_valid_rmse,GD_test_rmse)
            
            SGD_lr,SGD_epochs = 0.0005, 100
            #print("SGD")
            SGD_wt = SGD(train,weight,SGD_lr,SGD_epochs,precision)
            SGD_train_rmse ,SGD_valid_rmse, SGD_test_rmse = rmse(train,SGD_wt),rmse(validation,SGD_wt), rmse(test,SGD_wt)
            #print(SGD_train_rmse,SGD_valid_rmse, SGD_test_rmse)
        
            SGD_L1_wt = SGD_Lasso_reg(train,weight,SGD_lr,SGD_epochs,lambda_,precision)
            SGD_train_rmse,SGD_valid_rmse, SGD_test_rmse = rmse(train,SGD_L1_wt),rmse(validation,SGD_L1_wt), rmse(test,SGD_L1_wt)
            result_sgd_L1[count][1], result_sgd_L1[count][2], result_sgd_L1[count][3] = SGD_train_rmse,SGD_valid_rmse, SGD_test_rmse
            #print(SGD_train_rmse,SGD_valid_rmse, SGD_test_rmse)
        
            SGD_L2_wt = SGD_Ridge_reg(train,weight,SGD_lr,SGD_epochs,lambda_,precision)
            SGD_train_rmse ,SGD_valid_rmse, SGD_test_rmse = rmse(train,SGD_L2_wt),rmse(validation,SGD_L2_wt), rmse(test,SGD_L2_wt)
            result_sgd_L2[count][1], result_sgd_L2[count][2], result_sgd_L2[count][3] = SGD_train_rmse ,SGD_valid_rmse, SGD_test_rmse
            #print(SGD_train_rmse,SGD_valid_rmse, SGD_test_rmse)
            
            count+=1
        
        
        Plot_surface(train, x1, x2, i, GD_wt)
        result_gd_L1_df = pd.DataFrame(data = result_gd_L1, columns= gd_L1_column_title)
        result_gd_L2_df = pd.DataFrame(data = result_gd_L1, columns= gd_L2_column_title)
        result_sgd_L1_df = pd.DataFrame(data = result_sgd_L1, columns= sgd_L1_column_title)
        result_sgd_L2_df = pd.DataFrame(data = result_sgd_L2, columns= sgd_L2_column_title)
        print(result_gd_L1_df)
        print(result_gd_L2_df)
        print(result_sgd_L1_df)
        print(result_sgd_L2_df)
        
  
        
        
        
        
        