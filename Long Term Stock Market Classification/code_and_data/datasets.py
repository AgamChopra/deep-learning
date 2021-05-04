# datasets.py
# Author: Agamdeep S. Chopra, bit.ly/AgamChopra
# Last Updated: 05/01/2021

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing as pre
from matplotlib import pyplot as plt 
PATH = r'some_root_path'
#!!! Extracts the datasets
'''
def prep_data():
    df = pd.read_csv('PATH + '\Data\archive\Total.csv')
    df = df.fillna(0)
    df.to_excel(PATH + '\Data\archive\Total_Corrected')
'''

def data():
    x,y,z = pd.read_excel(PATH + '\Data\train.xlsx').values,pd.read_excel(PATH + '\Data\test.xlsx').values,pd.read_excel (PATH + '\Data\val.xlsx').values
    p = np.concatenate((x,y,z))
    for i in range(p[0,:].size):
        mx = max(p[:,i])
        mn = min(p[:,i])
        x[:,i] = (x[:,i]-mn)/(mx-mn)
        y[:,i] = (y[:,i]-mn)/(mx-mn)
        z[:,i] = (z[:,i]-mn)/(mx-mn)
    return x,y,z

def remove_outliers(arr, k):
    mu, sigma = np.mean(arr, axis=0), np.std(arr, axis=0, ddof=1)
    return arr[np.all(np.abs((arr - mu) / sigma) < k, axis=1)]

# Prints factors of a given number
def print_factors(x):
   print("The factors of",x,"are:")
   for i in range(1, x + 1):
       if x % i == 0:
           print(i)

# Returns correlation matrix
def corr(np_data):
    df = pd.DataFrame(np_data)
    corr = df.corr()
    return corr

#!!! Removes outliars for a given stdev from the dataset
def clean_data(stdev, train, test, val):
    return remove_outliers(train, stdev),remove_outliers(test, stdev),remove_outliers(val, stdev)

#!!! returns features with |Pearson Correlation| >= 0.6, can be used as filter for linear modeling
def filter_data(train,test,val):
    cor_max = corr(train)[:,-1]
    train_list = []
    test_list = []
    val_list = []
    for i in range(len(cor_max)):
        if cor_max[i] >= 0.6 or cor_max[i] <= -0.6:
            print("correlation of col", i,"with expectation col is:",cor_max[i])
            train_list.append(train[:,i])
            test_list.append(test[:,i])
            val_list.append(val[:,i])
    return np.array(train_list).T, np.array(test_list).T, np.array(val_list).T

#!!! Returns index of features that are the most influential for clustering the dataset
def Principal_Component_Analysis(train,k=20):
    scaled_train = pre.scale(train)
    pca = PCA()
    pca.fit(scaled_train)
    pca_train = pca.transform(scaled_train)
    per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)
    
    labels = ['PC'+str(x) for x in range(1,len(per_var)+1)]
    plt.bar(x=range(1,len(per_var)+1), height = per_var)
    plt.ylabel('% of explained variables')
    plt.xlabel('Principal Component (PC)')
    plt.title('Scree Plot')
    plt.show()
    
    pca_df = pd.DataFrame(pca_train, columns=labels)
    plt.scatter(pca_df.PC1, pca_df.PC2)
    plt.title('PCA Plot for PC 1 and 2')
    plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    plt.show()
    
    loading_scores = pd.Series(pca.components_[0])
    sorted_loading_scores = loading_scores.abs().sort_values(ascending = False)
    top_k_features = sorted_loading_scores[0:k].index
    return loading_scores[top_k_features].index,loading_scores[top_k_features].values

#!!! Maps  M -> N, where N<=M
#def JLT(data): 
#    return johnson_lindenstrauss_min_dim(n_samples=data[:,0].size, eps=0.5)

#!!! Extracts data -> (JLT/PCA/Corr <-> Outliars) -> Output
def processed_data(sd = 2, reduce_feature = "pearson-top", pca_k = 20):
    
    #Load dataset:
    train, test, val = data()
    
    #Outliar elimination using standard deviation:
    train, test, val = clean_data(sd, train, test, val)
    
    #Feature Engineering:
    if(reduce_feature == "pearson-top"):
        train, test, val = filter_data(train, test, val)
        train_x = train[:,:-1]
        train_y = train[:,-1]
        test_x = test[:,:-1]
        test_y = test[:,-1]
        val_x = val[:,:-1]
        val_y = val[:,-1]
        
    elif(reduce_feature == "pca"):
        train_x = train[:,:-1]
        train_y = train[:,-1]
        test_x = test[:,:-1]
        test_y = test[:,-1]
        val_x = val[:,:-1]
        val_y = val[:,-1]
        pca_index,_ = Principal_Component_Analysis(train_x,pca_k)
        train_x = train_x[:,pca_index]
        test_x = test_x[:,pca_index]
        val_x = val_x[:,pca_index]
        
    #Train set dimentionality reduction using Johnson Lindenstrauss Lemma:
    #elif(reduce_feature == "JL"):
    #    train = Johnson_Lindenstrauss_transform(train)
        
    return train_x, train_y, test_x, test_y, val_x, val_y

#!!! Splits the data into k batches for cross validation or mini-batch grad. desc., it is assumed that the data order is already randomized
def k_split(data,y,k=1):
    # returns the split array as [k, examples per k, features]
    print(data.shape,y.shape)
    index = 0
    leng = int(len(data[:,0]))
    arr = np.zeros((k,int(leng/k),data[0,:].size))  
    arry = np.zeros((k,int(leng/k),1))  
    for i in range (k):
        for j in range (int(leng/k)):
            for l in range (data[0,:].size):
                arr[i,j,l] = data[index,l]
                arry[i,j,0] = y[index]
            index += 1   
    return arr, arry