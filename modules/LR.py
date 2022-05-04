#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report
import seaborn as sns
import random

#Multiclass - distinguish between all three types
class LR:
    
    #Initialize
    def __init__(self):
        self.int = {}
        
    #Mean centre to standardise the data
#     def mean_centre(self, X_train):
#         return X_train-X_train.mean()
   
    #Sigmoid function - The Sigmoid Function transforms all its inputs between 0 and 1 
    def sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))
        
    #Hypothesis function -  It  limits the cost function between 0 and 1
    def hyp(self,X,theta):
        return self.sigmoid(np.dot(theta.T,X)) #61* 61*575
    
    #Feature_init adds a column of ones to our feature matrix
    def feature_init(self, X):
        return np.c_[np.ones((np.shape(X)[0],1)),X]
    
    def random(self, X, Y):
        rand=random.randrange(X.shape[0])
        x=X[rand]
        y=Y[rand] 
        return x, y

    #Gradient Descent updates the theta
    def gradient_descent(self, X, Y, alpha, iterations, lamb):
        X1=self.feature_init(X)
        #Initialize the parameters to 0
        theta=np.zeros(X1.shape[1]) 
        for i in range(iterations):
            x, y =self.random(X1, Y)
            h_x=self.hyp(x.T,theta)
            
            weights=np.zeros(x.shape[0])
            
            for j in range(x.shape[0]):                  
                weights[j]= np.dot(x[j], h_x-y) #+ 2*lamb*theta[j]
            
            L2=1-alpha*lamb/X.shape[1]
            theta=theta*L2 -(alpha*weights/X.shape[1])
            #theta=theta-alpha*weights
                
        #return theta(1-(alpha*lamb/X.shape[0]))-(alpha*weights/X.shape[0])
        #return theta*L2 -(alpha*weights/X.shape[1])
        return theta
        

    def fit(self, X, Y):
        #X=self.mean_centre(X)
        all_theta={}
        alpha=1.25
        iterations=10000
        lamb=0.01
        #Multiclass - Set one class to 1 and all others to 0
        glioma=np.unique(Y)
        for glioma_type in glioma:
            Y1=Y.copy() 
            Y1=np.where(Y1==glioma_type, 1, 0)
            theta=self.gradient_descent(X, Y1, alpha, iterations, lamb)
            all_theta.update({glioma_type:theta})   
        self.int=all_theta

    def predict(self,X):
        #X=self.mean_centre(X)
        X=self.feature_init(X) #575*60+1
        Y=[]
        h_x_list=[]      
        glioma=[]
        
        for glioma_type,theta in self.int.items():                  
            h_x_list.append(self.hyp(X.T,theta)) #61*575
            glioma.append(glioma_type)
        
        temp_df = pd.DataFrame(h_x_list).T
        temp_df['y_pred'] = temp_df.apply(lambda x: glioma[np.argmax(x)], axis = 1)
        #Y=np.apply_along_axis(lambda x:glioma[np.argmax(x)],0,h_x_list)
        return temp_df['y_pred'].to_numpy()
        #return Y
        
    def cross_valid(self,X,y):
        master_dict = {'accuracy':[],'precision':[],'recall':[],'f1':[]}
        kf = KFold()
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = self
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test,y_pred,output_dict=True)
            master_dict['accuracy'].append(report['accuracy'])
            master_dict['precision'].append(report['macro avg']['precision'])
            master_dict['recall'].append(report['macro avg']['recall'])
            master_dict['f1'].append(report['macro avg']['f1-score'])
        
        result ={}
        result['accuracy'] = np.mean(master_dict['accuracy'])
        result['precision'] = np.mean(master_dict['precision'])
        result['recall'] = np.mean(master_dict['recall'])
        result['f1'] = np.mean(master_dict['f1'])
        return result

