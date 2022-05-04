#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve


# In[2]:


class KNN:
    def __init__(self):
        self.init = {}
        
    #euclidean_dist calculates the euclidian distance between each sample 
    def euclidean_dist(self,sample1, sample2):
        s1 = sample1.to_numpy()
        s2 = sample2.to_numpy()
        return np.power(np.sum(np.power(np.subtract(s1,s2), 2)), 0.5)

    #appending all distances calculated to an array
    def get_all_distances(self,train, test):
        d = []
        for i in range(len(test.index)):
            vals = []
            for j in range(len(train.index)):
                vals.append(self.euclidean_dist(test.iloc[i][1:], train.iloc[j][1:]))
            d.append(vals)
        #Converting the array to numpy array
        dnp = np.array(d)
        return dnp
    
    # get_k_nearest neighbors obtains the k nearest neighbors according to the euclidean distance calculated
    def get_K_nearest_neighbors(self,dnp, K):
        neighbors = dnp.argsort(axis=-1)[:,:K]
        return neighbors.reshape((-1,K))
    
    def predict_label(self,neighbors, train_y):
        pred_labels = []
        for row in neighbors:
            labels = [train_y.iloc[x] for x in row]
            majority = (max(set(labels), key = labels.count))
            pred_labels.append(majority)
        return pred_labels

    def perform_KNN(self,train_X, test_X, train_y, K):
        dnp = self.get_all_distances(train_X, test_X)
        neighbors = self.get_K_nearest_neighbors(dnp, K)
        return self.predict_label(neighbors, train_y)

    def cross_validate(self,X,y,K):
        master_dict = {'accuracy':[],'precision':[],'recall':[],'f1':[]}
        kf = KFold()
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]
            y_pred = self.perform_KNN(X_train,X_test,y_train,K)
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


# In[ ]:




