from pyexpat import model
import numpy as np
from sklearn.metrics import confusion_matrix

"""
Difference of false postive rate between previleaged group and unprevileaged group(DEO)

Params:
--------
y_pred(np.array): prediction output of model
y_label(np.array): label of input
x(np.array): input
group_index(int): index of group in feature vector
feature_p: class value of previleaged group in feature vector
feature_un: class value of unprevileaged group in feature vector
--------

return value of DEO

"""
def deo(y_pred,y_label,x,group_index=9,feature_p='0',feature_un='1'):
    tn_un, fp_un, _, _ = confusion_matrix(y_label[x[:,group_index] == feature_un], y_pred[x[:,group_index] == feature_un]).ravel()
    tn_p, fp_p, _, _ = confusion_matrix(y_label[x[:,group_index] == feature_p], y_pred[x[:,group_index] == feature_p]).ravel()

    fpr_un = fp_un/(fp_un+tn_un)
    fpr_p = fp_p/(fp_p+tn_p)

    return -np.abs(fpr_un - fpr_p)

"""
DEO augumented (using lamda to assign different ratio between two groups)

deo_aug = deo + (FPR_un + lamda * FPR_p)

Params:
--------
y_pred(np.array): prediction output of model
y_label(np.array): label of input
x(np.array): input
feature_p: class value of previleaged group in feature vector
feature_un: class value of unprevileaged group in feature vector
group_index(int): index of group in feature vector
lamda(float): regularization term for privileaged group
---------

return value of augumented deo

"""
def deo_aug(y_pred,y_label,x,group_index=9,feature_p='0',feature_un='1',lamda=0.6):
    tn_un, fp_un, _, _ = confusion_matrix(y_label[x[:,group_index] == feature_un], y_pred[x[:,group_index] == feature_un]).ravel()
    tn_p, fp_p, _, _ = confusion_matrix(y_label[x[:,group_index] == feature_p], y_pred[x[:,group_index] == feature_p]).ravel()

    fpr_un = fp_un/(fp_un+tn_un)
    fpr_p = fp_p/(fp_p+tn_p)

    return -np.abs(fpr_un - fpr_p) - (fpr_un+lamda*fpr_p)

"""
m:model
x: input
y_label: label of of input
"""
def false_positive(m,x,y_label,):

    y_pred = m.predict(x) 
    tn_aa, fp_aa, _, _ = confusion_matrix(y_label[x[:,9] == '1'], y_pred[x[:,9] == '1']).ravel()
    tn_w, fp_w, _, _ = confusion_matrix(y_label[x[:,9] == '0'], y_pred[x[:,9] == '0']).ravel()

    fpr_aa = fp_aa/(fp_aa+tn_aa)
    fpr_w = fp_w/(fp_w+tn_w)

    return -(fpr_aa,fpr_w,np.abs(fpr_aa - fpr_w))

