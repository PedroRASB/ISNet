#evaluation funcitons for multi-class, single-label problems
from sklearn.metrics import multilabel_confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import compare_auc_delong_xu as DeLongAUC
import os
import sklearn

def AddImplicitClass(out,th=0.5):
    #makes implicit class explicit, by adding an extra class to out
    #implicitClass:consider that no class is present (empty) if max(out)<th
    #labels should include the empty class as the last one and be indexesnumClasses=numClasses+1
    
    newOut=torch.zeros((out.shape[0],out.shape[1]+1)).type_as(out)#add extra class
    newOut[:,:-1]=out
    MaxConfidence=torch.max(out,dim=1,keepdim=True)[0]#maximum confidence in any class, per sample
    b=2*th#bias to ensure empty if max(out)<th
    EmptyConfidence=torch.ones(MaxConfidence.shape).type_as(MaxConfidence)*b-MaxConfidence
    newOut[:,-1]=EmptyConfidence.squeeze()#add confidence scores for the empty class
    return newOut
    
def FindImplicitClassTh(out,labels):
    #Find th that generates the best F1-Score in validation dataset
    F1s=[]
    ths=[]
    maximum=int(torch.max(out).item()*1000)
    for th in list(range(maximum)):
        th=th/1000
        P=PerformanceMatrix(ConfusionMatrix(out,labels,implicitClass=True,th=th))
        F1s.append(P[-1,-2])
        ths.append(th)
    th=ths[np.argmax(F1s)]
    return th
    

def ConfusionMatrix(out,labels,implicitClass=False,th=0.5):
    numClasses=out.shape[-1]
    if(implicitClass):
        newOut=AddImplicitClass(out=out,th=th)
        numClasses=numClasses+1
    else:
        newOut=out
        
    M=np.zeros((numClasses,numClasses))#rows=real class, columns=predicted class

    try:
        labels=labels.squeeze(1)
    except:
        pass
    predictions=torch.argmax(newOut,dim=1)
    
    for i,_ in enumerate(labels,0):
        l=int(labels[i].item())#real class
        p=int(predictions[i].item())#predicted class
        M[l][p]=M[l][p]+1
        
    return (M)

def Acc(out,labels):
    M=ConfusionMatrix(out,labels)
    accuracy=np.sum([M[i][i] for i in range(len(M))])/np.sum(M)
    return accuracy



def PerformanceMatrix (matrix):
    #matrix:Confusion matrix
    numClasses=len(matrix)
    M=matrix
    P=np.zeros((numClasses+1,3))
    
    i=0
    #P rows:classes, mean. P columns:precision, recall, F1, specificity.
    for c in range(numClasses):
        #False negatives:
        FN=np.sum([M[c][i] if (i!=c) else 0 for i in range(numClasses)])
        #false positives:
        FP=np.sum([M[i][c] if (i!=c) else 0 for i in range(numClasses)])
        #true negatives:
        TN=np.sum([M[i][j] if (i!=c and j!=c) else 0 \
                   for i in range(numClasses) for j in range(numClasses)])
        #true positives:
        TP=M[c][c]
        
        P[c][0]=TP/(TP+FP+1e-6)#precision
        P[c][1]=TP/(TP+FN+1e-6)#recall
        P[c][2]=2*(P[c][0]*P[c][1])/(P[c][0]+P[c][1]+1e-6)#F1-score
        #P[c][3]=TN/(TN+FP+1e-6)#specificity
        
    #means
    for k in range(3):
        P[numClasses][k]=np.mean([P[i][k] for i in range(numClasses)])
        
    #round
    P=np.around(P,decimals=3)
    return(P)