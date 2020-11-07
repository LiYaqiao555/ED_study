# -*- coding: utf-8 -*-

import iaml01cw2_helpers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#load datasets from Fashion-MNIST
Xtrn, Ytrn, Xtst, Ytst=iaml01cw2_helpers.load_FashionMNIST("Fashion_MNIST")
'''
4 data normalization steps
'''
#Step 1: back up data
Xtrn_orig=Xtrn
Xtst_orig=Xtst

#Step 2: Divide data by 255.0
Xtrn=Xtrn/255.0
Xtst=Xtst/255.0

#Step 3: calculate mean of Xtrn for each dimension
Xmean=Xtrn.mean(0)

#Step 4: normalization 
Xtrn_nm=Xtrn-Xmean
Xtst_nm=Xtst-Xmean

'''
Question 1.1
report the first 4 elements for the first training sample and last traininig sample:
Xtrn_nm[0,0:4]
Xtrn_nm[-1,0:4]
'''


#Question 1.2
def sample_select(class_num):
    #finding the Xtrn data corresponding to specific class
    Xtrn_class=Xtrn[Ytrn==class_num]               
    #Calculating Euclidean distance for every example                         
    Xtrn_class_Euc=Xtrn_class-Xtrn_class.mean(0)
    Xtrn_class_Euc=np.sum(np.square(Xtrn_class_Euc),axis=1)
    #sort the Euclidean distance examples to find two closest samples and two furthest samples
    Xtrn_class_Euc_sort=np.sort(Xtrn_class_Euc)
    #two closest samples
    closest_samples_1=Xtrn_class[int(np.argwhere(Xtrn_class_Euc==Xtrn_class_Euc_sort[0]))]
    closest_samples_2=Xtrn_class[int(np.argwhere(Xtrn_class_Euc==Xtrn_class_Euc_sort[1]))]
    #two furthest samples
    furthest_samples_1=Xtrn_class[int(np.argwhere(Xtrn_class_Euc==Xtrn_class_Euc_sort[-1]))]
    furthest_samples_2=Xtrn_class[int(np.argwhere(Xtrn_class_Euc==Xtrn_class_Euc_sort[-2]))]
    mean_sample=Xtrn_class.mean(0)
    
    #reshape samples
    closest_samples_1=closest_samples_1.reshape((28,28))
    closest_samples_2=closest_samples_2.reshape((28,28))
    furthest_samples_1=furthest_samples_1.reshape((28,28))
    furthest_samples_2=furthest_samples_2.reshape((28,28))
    mean_sample=mean_sample.reshape((28,28))
    
    return mean_sample,closest_samples_1,closest_samples_2,furthest_samples_2,furthest_samples_1

#find the index of corresponding samples
def sample_selected_index(cls_1,cls_2,fst_2,fst_1):
    #flatten the shape of 28x28 into 784
    cls_1=cls_1.flatten()
    cls_2=cls_2.flatten()
    fst_2=fst_2.flatten()
    fst_1=fst_1.flatten()
    
    cls_1_index=np.where(np.all(Xtrn==cls_1,axis=1))
    cls_2_index=np.where(np.all(Xtrn==cls_2,axis=1))
    fst_2_index=np.where(np.all(Xtrn==fst_2,axis=1))
    fst_1_index=np.where(np.all(Xtrn==fst_1,axis=1))
    return cls_1_index,cls_2_index,fst_2_index,fst_1_index

def plot_grid():
   
    for i in range(0,50):
        #get the five spcific samples in each class
        mean,cls_1,cls_2,fst_2,fst_1=sample_select(i//5)
        #get the samples' index
        cls_1_index,cls_2_index,fst_2_index,fst_1_index=sample_selected_index(cls_1,cls_2,fst_2,fst_1)
        #store them in tuple
        store_samples=[mean,cls_1,cls_2,fst_2,fst_1]
        sotre_index=['mean',int(cls_1_index[0]),int(cls_2_index[0]),int(fst_2_index[0]),int(fst_1_index[0])]
        
        plt.subplot(10,5,i+1)
        plt.rcParams['axes.titlepad'] = -10
        plt.imshow(store_samples[i%5],cmap='gray_r')
        plt.title('class%d,index:%s'%(i//5,sotre_index[i%5]),fontsize=5)
        plt.axis('off') 
    #save the picture     
    plt.savefig('filename.png', dpi = 300)
        
        
#Question 1.3
def first_five_EV():
    pca=PCA()
    pca.fit(Xtrn_nm)    
    print(pca.explained_variance_[0:5])

#Question 1.4
def cumulative_EVR_plot():
    pca=PCA()
    pca.fit(Xtrn_nm)  
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    
#Question 1.5

def pc_draw():
    pca=PCA()
    pca.fit(Xtrn_nm)  
    PC_components=pca.components_ 
    for i in range(10):
        principal_components=PC_components[i].reshape((28,28))
        plt.subplot(2,5,i+1)
        plt.imshow(principal_components,cmap='gray_r')
        plt.axis('off')
    plt.savefig('Q1.5.png', dpi = 300)
        
    
#Question 1.6
def RMSE_calculate(class_num,K):
    #finding the Xtrn data corresponding to specific class
    Xtrn_class=Xtrn_nm[Ytrn==class_num]   
    pca=PCA(n_components=K)
    #apply dimensionality reduction with PCA to the first sample in the class
    transformed_sample=pca.fit_transform(Xtrn_class) 
    #reconstruct the sample from the dimensionality-reduced sample
    reconstruct_sample=pca.inverse_transform(transformed_sample)
    
    #Calculate the RMSE
    from sklearn.metrics import mean_squared_error
    import math
    mse=mean_squared_error(reconstruct_sample[0],Xtrn_class[0])
    rmse = math.sqrt(mse)
    
    return rmse
def class_K_cal():
    for class_num in range(10):
        for K in [5,20,50,200]:
            print('K:',K,' class number: ',class_num,' RMSE: "%.3f'%RMSE_calculate(class_num, K))
            print(class_num,'&')
            
#Question 1.7
def reconst_samples(class_num,K):
    #finding the Xtrn data corresponding to specific class
    Xtrn_class=Xtrn_nm[Ytrn==class_num] 
    pca=PCA(n_components=K)
    #apply dimensionality reduction with PCA to the first sample in the class
    transformed_sample=pca.fit_transform(Xtrn_class) 
    #reconstruct the sample from the dimensionality-reduced sample
    reconstruct_sample=pca.inverse_transform(transformed_sample)
    #add Xmean to each reconstructed sample
    reconstruct_sample=reconstruct_sample+Xmean
    reconstruct_output=reconstruct_sample[0].reshape((28,28))
    return reconstruct_output  

def reconst_plot():
    i=0
    for class_num in range(10):        
        for K in [5,20,50,200]:
            i=i+1
            plt.subplot(10,4,i)
            plt.rcParams['axes.titlepad'] = -10
            plt.imshow(reconst_samples(class_num, K),cmap='gray_r')
            plt.title('class:%d K: %d' %(class_num,K),fontsize=5)            
            plt.axis('off')
            
    plt.savefig('Q1.7.png', dpi = 300)
    
#Question 1.8
def two_d_PCA_plot():
    plt.figure()
    for class_num in range(10):
    
        Xtrn_nm_class=Xtrn_nm[Ytrn==class_num]          
        pca=PCA(n_components=2)        
        two_d_PCA_samples=pca.fit_transform(Xtrn_nm_class)           
        plt.scatter(two_d_PCA_samples[:,0],two_d_PCA_samples[:,1],marker='o',cmap=plt.cm.coolwarm,s=2,label='class:%d'%class_num)
        plt.legend(loc='best')
    plt.xlabel('principal component 1')
    plt.ylabel('principal componet 2')
    plt.show()


#Question 1.9

def DCT_plot():
    plt.figure(figsize=(14,7))
    D=784
    c_ki=np.zeros((3,784))
    #calculate basis vector
    for k in range(1,3+1):
        for i in range(1,D+1):
            c_ki[k-1,i-1]=np.cos((np.pi*(k-1)*(2*i-1)/(2*D)))
    
    #plot the image of c_k for k=1,2,3
    for k in range(3):
        basis_vec=c_ki[k,:].reshape((28,28))
        plt.subplot(1,3,k+1)
        plt.imshow(basis_vec,cmap='gray_r' )
        
                            
#Question 1.10
def DCT_PCA_compare():
    D=784
    c_ki=np.zeros((784,784))
    #calculate basis vector 
    for k in range(1,785):
        for i in range(1,785):
            c_ki[k-1,i-1]=np.cos((np.pi*(k-1)*(2*i-1)/(2*D)))
    #Apply DCT and calculate the variance
    z_k=np.dot(Xtrn_nm,c_ki)
    var_zk=np.zeros((1,784))
    for j in range(z_k.shape[1]):
        var_zk[:,j]=np.var(z_k[:,j])
    #calculate cumulative explained variance
    var_zk=var_zk/np.sum(var_zk)
    
    #, plot the cumulative explained variance ratio for both DCT and PCA
    pca=PCA()
    pca.fit(Xtrn_nm)  
    plt.plot(np.cumsum(pca.explained_variance_ratio_),c='blue',label='PCA')
    plt.plot(np.cumsum(var_zk),c='red',label='DCT')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.legend(loc='best')
    plt.show()
        
                        
      
        
    
               
    

    
