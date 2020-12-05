# -*- coding: utf-8 -*-

import iaml01cw2_helpers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
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

#Question 2.1
#report classification acc and confusion matrix for test set
def Question2_1():
    ml_LR=LogisticRegression()
    ml_LR.fit(Xtrn_nm,Ytrn)
    #mean classification accuracy
    tst_acc=ml_LR.score(Xtst_nm,Ytst)
    Y_pred=ml_LR.predict(Xtst_nm)
    #confusion matrix
    confusion_mat=confusion_matrix(Ytst, Y_pred)
    # return Y_pred,confusion_mat
    #For confusion matrix table form in Latex
    # for i in range(10):
    #     for j in range(10):
    #         print('&',confusion_mat[i,j],end='')
    #     print(end='\n')
            

#Question 2.2
def Question2_2():
    clf=SVC(kernel='rbf', C = 1.0,gamma='auto')          
    clf.fit(Xtrn_nm,Ytrn)        
    mean_acc=clf.score(Xtst_nm,Ytst)
    Y_pred=clf.predict(Xtst_nm)
    #confusion matrix
    confusion_mat=confusion_matrix(Ytst,Y_pred)
    #For confusion matrix table form in Latex
    # for i in range(10):
    #     for j in range(10):
    #         print('&',confusion_mat[i,j],end='')
    #     print(end='\n') 
    return confusion_mat,mean_acc       
    
#Question 2.3
def Question2_3():
    pca=PCA(n_components=2)
    pca.fit(Xtrn_nm) 

    # Calculate the std for the first two principal components
    # standard deviations=variance**0.5  
    pc_std_1=pca.explained_variance_[0]**0.5
    pc_std_2=pca.explained_variance_[1]**0.5
    x=np.linspace(-5*pc_std_1,5*pc_std_1,100)
    y=np.linspace(-5*pc_std_2,5*pc_std_2,100)
    xx,yy=np.meshgrid(x,y) 
    
    # Finding the points in original vector space
    inverse_transformed=pca.inverse_transform(np.c_[xx.flatten(),yy.flatten()])
    # Apply multinomial logistic regression 
    ml_LR=LogisticRegression()
    ml_LR.fit(Xtrn_nm,Ytrn)
    Y_pred=ml_LR.predict(inverse_transformed)
    
    # Visualise the decision regions 
    Y_pred= Y_pred.reshape((len(x),len(y)))
    levels=np.linspace(0,10,11)
    plt.contourf(xx,yy,Y_pred,levels,cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.savefig('Q2.3.png', dpi = 300)
    plt.show()
    
#Question 2.4
def Question2_4():
    pca=PCA(n_components=2)
    pca.fit(Xtrn_nm)

    # Calculate the std for the first two principal components
    # standard deviations=variance**5    
    pc_std_1=pca.explained_variance_[0]**0.5
    pc_std_2=pca.explained_variance_[1]**0.5
    x=np.linspace(-5*pc_std_1,5*pc_std_1,100)
    y=np.linspace(-5*pc_std_2,5*pc_std_2,100)
    xx,yy=np.meshgrid(x,y)    
    
    # Finding the points in original vector space
    inverse_transformed=pca.inverse_transform(np.c_[xx.flatten(),yy.flatten()])
    
    #Apply SVM classifier
    clf=SVC(kernel='rbf', C = 1.0,gamma='auto')          
    clf.fit(Xtrn_nm,Ytrn) 
    Y_pred=clf.predict(inverse_transformed)
    
    # Visualise the decision regions 
    Y_pred= Y_pred.reshape((np.shape(xx)))
    levels=np.linspace(0,10,11)
    plt.contourf(xx,yy,Y_pred,levels,cmap=plt.cm.coolwarm)
    plt.colorbar()
    plt.savefig('Q2.4.png', dpi = 300)
    plt.show()
    
#Question 2.5
#Create parameter Xsmall to pick first 1000 traning samples from each class
def Cross_Validation_trn_data():
    X_small=Xtrn_nm[Ytrn==0][0:1000]
    Y_small=Ytrn[Ytrn==0][0:1000]
    for class_num in range(1,10):
        Xtrn_nm_class=Xtrn_nm[Ytrn==class_num][0:1000]
        Ytrn_class=Ytrn[Ytrn==class_num][0:1000]
        X_small=np.concatenate((X_small,Xtrn_nm_class))
        Y_small=np.concatenate((Y_small,Ytrn_class))
    return X_small,Y_small 
Xsmall,Ysmall=Cross_Validation_trn_data()

def Question2_5():
    from sklearn.model_selection import cross_val_score
    #use 10 values spaced equally log space between 10^-2 to 10^3
    penalty_C=np.logspace(-2,3,10)
    mean_clf_accuracy=[]
    for i in penalty_C:
        clf=SVC(kernel='rbf', C = i,gamma='auto')          
        clf.fit(Xsmall,Ysmall)
        score=cross_val_score(clf,Xsmall,Ysmall,cv=3)
        
        average_score=(score[0]+score[1]+score[2])/3
        print(score)
        print('mean score:', average_score)
        
        mean_clf_accuracy.append(average_score)
        
    #Plot the mean cross-validated classification accuracy against the regularisation parameter  
    fig, ax = plt.subplots()    
    plt.scatter(penalty_C,mean_clf_accuracy)
    ax.set_xscale('log')
    plt.xlabel('regularisation parameter')
    plt.ylabel('mean cross-validated clf accuracy') 
    plt.xticks(penalty_C)
    plt.savefig('Q2.5.png', dpi = 300)
    plt.show()
    



#Question 2.6
def Question2_6():
    #training SVM clf on training set using optimal value of C
    #from the observation of Question2.5,the optimal value of C is:
    C_penalty=np.logspace(-2,3,10)[6]
    clf=SVC(kernel='rbf', C = C_penalty,gamma='auto')          
    clf.fit(Xtrn_nm,Ytrn) 
    #report clf accuracy on trn set and test set
    clf_acc_trn=clf.score(Xtrn_nm,Ytrn)
    clf_acc_tst=clf.score(Xtst_nm,Ytst)
    return clf_acc_trn,clf_acc_tst

        
