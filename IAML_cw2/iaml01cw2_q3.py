# -*- coding: utf-8 -*-

import iaml01cw2_helpers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import math

Xtrn,Ytrn,Xtst,Ytst = iaml01cw2_helpers.load_CoVoST2('data')

#Question 3.1
def kmeans_clust(): 
    Kmeans=KMeans(n_clusters=22, random_state=1).fit(Xtrn)
    #Report the sum of squared distances of samples to their closest cluster centre
    print(Kmeans.inertia_)
    
    #the number of samples for each cluster.    
    mydict={i:np.count_nonzero(Kmeans.labels_==i) for i in range(22)}
    print(mydict)
    

def Question_3_2():
    #calculate mean vector for each language 
    Xtrn_mean=np.zeros((22,26))  
    for i in range(22):
        Xtrn_mean[i]=np.mean(Xtrn[Ytrn==i],axis=0)
        
    #Applying PCA to Xtrn_mean 
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2)
    Xtrn_mean_pca=pca.fit_transform(Xtrn_mean)
    
    #Applying PCA to cluster centers
    Kmeans=KMeans(n_clusters=22, random_state=1).fit(Xtrn)
    Kmean_cc_pca=pca.transform(Kmeans.cluster_centers_)
    n=np.arange(22)
    
    
    #Plot the Xtrn_mean and cluster centres on the same figure
    plt.figure(figsize=(14,8))
    #for Xtrn_mean
    
    plt.scatter(Xtrn_mean_pca[:,0],Xtrn_mean_pca[:,1],label='mean vector',s=70,c='red')
    plt.legend(loc='upper left')
    for i,text in enumerate(n):
        plt.annotate(text,(Xtrn_mean_pca[i,0],Xtrn_mean_pca[i,1]))
        
    #for cluster centres
    
    plt.scatter(Kmean_cc_pca[:,0],Kmean_cc_pca[:,1],label='cluster centres',marker='s',s=70,c='blue')
    plt.legend(loc='upper left')
    for i,text in enumerate(n):
        plt.annotate(text,(Kmean_cc_pca[i,0],Kmean_cc_pca[i,1]))
    
    plt.savefig('Q3.2.png', dpi = 300)
    plt.show()

    
#Question 3.3
def Question3_3():
    #calculate mean vector for each language 
    Xtrn_mean=np.zeros((22,26))  
    for i in range(22):
        Xtrn_mean[i]=np.mean(Xtrn[Ytrn==i],axis=0)
    # carry out hierarchical clustering with the Ward’s linkage 
    mergings=linkage(Xtrn_mean,method='ward')
    # display the dendrogram 
    dendrogram(mergings,orientation='right')
    plt.savefig('Q3.3.png', dpi = 300)
    plt.show()
    
#Question 3.4
def Question3_4():
    #Apply kmeans clustering 
    Kmeans=KMeans(n_clusters=3, random_state=1)
    Kmeans.fit(Xtrn[Ytrn==0])
    vector=Kmeans.cluster_centers_
    for i in range(1,22):
        Kmeans.fit(Xtrn[Ytrn==i])
        centers=Kmeans.cluster_centers_
        vector=np.row_stack((vector,centers))
    
    label=[]
    for i in range(0,22):
        for j in range(0,3):
            label.append(str(i))
    #carry out hierarchical clustering with ’ward’, ’single’, and ’complete’ linkage method
    plt.figure(figsize=(15,10))
    mergings=linkage(vector,method='ward')
    dendrogram(mergings,orientation='right',labels=label)
    plt.title('ward linkage')
    plt.savefig('Q3_4_1.png', dpi = 300)
    
    plt.figure(figsize=(15,10))
    mergings=linkage(vector,method='single')
    dendrogram(mergings,orientation='right',labels=label)
    plt.title('single linkage')
    plt.savefig('Q3_4_2.png', dpi = 300)
     
    plt.figure(figsize=(15,10))
    mergings=linkage(vector,method='complete')
    dendrogram(mergings,orientation='right',labels=label) 
    plt.title('complete linkage')
    plt.savefig('Q3_4_3.png', dpi = 300)
    plt.show()
    
    
#Question 3.5
def Question3_5():
    # n_components= 1, 3, 5, 10, 15.    covariance_type='diag'  or 'full'
    cov_type=['diag','full']
    K=[1,3,5,10,15]
    mean_LL_trn=[]
    mean_LL_tst=[]
    Lan0_trn=Xtrn[Ytrn==0]
    Lan0_tst=Xtst[Ytst==0]
    for i in cov_type:
        for j in K:
            Gaussian_M=GaussianMixture(n_components=j,covariance_type=i)
            Gaussian_M.fit(Lan0_trn)     
            # the per-sample average log-likelihood on the training data and test data for Language 0

            mean_LL_trn.append(round(Gaussian_M.score(Lan0_trn),3))
            mean_LL_tst.append(round(Gaussian_M.score(Lan0_tst),3))
    #plot the result in a graph
    name_list = ['diag K=1','diag K=3','diag K=5','diag K=10','diag K=15',\
                 'full K=1','full K=3','full K=5','full K=10','full K=15']
    num_list = mean_LL_trn
    num_list1 = mean_LL_tst
    x =list(range(len(num_list)))
    total_width, n = 0.8, 2
    width = total_width / n
    
    plt.figure(figsize=(14,8)) 
    plt.bar(x, num_list, width=width, label='per-sample average train log-likelihood',fc = 'y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='per-sample average test log-likelihood',tick_label = name_list,fc = 'r')
    plt.legend()
    plt.savefig('Q3_5.png', dpi = 300)
    return mean_LL_trn,mean_LL_tst

    
# Question 3.6
def SimpleGMM_train(X, n_components):
    # apply Kmeans clustering
    Kmeans=KMeans(n_clusters=n_components, random_state=1).fit(X)
    # mean vectors (ndarray, shape (n_components, n_features))
    Xmean=Kmeans.cluster_centers_
       
    labels=Kmeans.labels_
    # a vector of diagonal elements of covariance matrix 
    diagonal_elements=np.zeros((n_components,X.shape[1]))
    for i in range(n_components):
        diagonal_elements[i,:]=np.diag(np.cov(X[labels==i].T))
         
    # weights of mixture components
    weights=np.zeros((n_components))
    for j in range(n_components):       
        weights[j]=np.count_nonzero(labels==j)/X.shape[0]
    return Xmean,diagonal_elements,weights
    





def SimpleGMM_eval(X, Ms, Dcovs, Pk):
    n_components=len(Ms)
    Kmeans=KMeans(n_clusters= n_components, random_state=1).fit(X)
    labels=Kmeans.labels_
    
    loglikelihoods=[]
    for i in range(n_components):
        loglikelihoods.append(Pk[labels[i]]*multivariate_normal.pdf(X, mean=Ms[i], cov=Dcovs[i]))
            
    p=pd.DataFrame(loglikelihoods)
    q=np.array(p.describe().loc['mean'])   
    lh=q*n_components
    
    Log_L_samples=[]
    for w in range(len(lh)):
        Log_L_samples.append(np.log(lh[w]))
          
    return Log_L_samples




#Question 3.7
def Question3_7():
    Lan0_trn=Xtrn[Ytrn==0]
    
    Xmean_1,diagonal_elements_1,weights_1=SimpleGMM_train(Lan0_trn,1)
    Xmean_5,diagonal_elements_5,weights_5=SimpleGMM_train(Lan0_trn,5)
    Xmean_10,diagonal_elements_10,weights_10=SimpleGMM_train(Lan0_trn,10)
    
    
    return weights_1,weights_5,weights_10

#Question 3.8
def Question3_8():
    Lan0_trn=Xtrn[Ytrn==0]
    Lan0_tst=Xtst[Ytst==0]
    K=[1,3,5,10,15]
    avg_LL_trn=[]
    avg_LL_tst=[]
    for i in K:
        Xmean_trn,diagonal_elements_trn,weights_trn=SimpleGMM_train(Lan0_trn,i)
        avg_LL_trn.append(np.mean(SimpleGMM_eval(Lan0_trn, Xmean_trn, diagonal_elements_trn, weights_trn)))
        
        Xmean_tst,diagonal_elements_tst,weights_tst=SimpleGMM_train(Lan0_trn,i)
        avg_LL_tst.append(np.mean(SimpleGMM_eval(Lan0_tst, Xmean_tst, diagonal_elements_tst, weights_tst)))
                          
    return avg_LL_trn,avg_LL_tst
    
    
    