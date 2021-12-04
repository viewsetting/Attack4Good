import pandas as pd
import numpy as np
from IPython.display import Markdown, display
from matplotlib.lines import Line2D

import time
import tqdm
import warnings
from scipy import integrate

from scipy import stats
random_state = 999
from copy import deepcopy



## General importsÇ
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from matplotlib.colors import ListedColormap


## SKLearn imports
from sklearn import linear_model, svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA

import math
import numpy as np
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data

## AIF360 imports
from aif360.datasets import BinaryLabelDataset, StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools


#SEC_ML imports
from secml.data.c_dataset import CDataset
from secml.ml.classifiers import CClassifierSVM, CClassifierLogistic
from secml.ml.kernels import CKernelRBF, CKernelLinear
from secml.ml.peval.metrics import CMetricAccuracy
from secml.data.splitter import CDataSplitterKFold

# Poisoning attacks
from secml.adv.attacks import CAttackPoisoningSVM
from secml.adv.attacks.poisoning.c_attack_poisoning_logistic_regression import CAttackPoisoningLogisticRegression


## AIF360 imports


def calculate_disparate_impact(y,sensible_att_vals, privileged_classes=1, favorable_output=1, verbose=False):
    
    privileged = y[sensible_att_vals == privileged_classes]     # white guys 
    unprivileged = y[sensible_att_vals != privileged_classes]   
    
    unprivileged_favorable = unprivileged[unprivileged==favorable_output]
    privileged_favorable = privileged[privileged==favorable_output]
    
    n1 =  (len(unprivileged_favorable)/ len(unprivileged))
    n2 = (len(privileged_favorable)/ len(privileged))
    print(privileged.shape, unprivileged.shape)
    
    if verbose:
        print("\tUnprivileged favorable1: ", n1)
        print("\tPrivileged favorable2: ", n2)
     
    disparate_impact = n1 - n2 #(max(n2,0.1)) 
    return disparate_impact

   

def get_data2():
    protected_attribute_used = 1 # 1, 2
    if protected_attribute_used == 1:     
        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        dataset_orig = load_preproc_data_compas(['sex'])
    else:
        privileged_groups = [{'race': 1}]
        unprivileged_groups = [{'race': 0}]
        dataset_orig = load_preproc_data_compas(['race'])

    optim_options = {
        "distortion_fun": get_distortion_compas,
        "epsilon": 0.05,
        "clist": [0.99, 1.99, 2.99],
        "dlist": [.1, 0.05, 0]
        }
    #random seed
    np.random.seed(1)

    dataset_aif360 = dataset_orig
    
    # Split into train, validation, and test
    dataset_orig_train, dataset_orig_vt = dataset_aif360.split([0.8], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    
    dataset_aif360 = dataset_orig_vt.copy()
    
    SENSIBLE_ATT_INDEX = dataset_orig.feature_names.index(dataset_orig.protected_attribute_names[0])

    ## Correcting labels assignation
    if dataset_aif360.unfavorable_label != 0:
        Y = dataset_aif360.labels
        # Y[Y == dataset_aif360.unfavorable_label] = 1
        # Y[Y == dataset_aif360.favorable_label] = -1
       
        # # original 
        Y[Y == dataset_aif360.unfavorable_label] = -1   
        Y[Y == dataset_aif360.favorable_label] = 1
        Y[Y == -1] = 0

        dataset_aif360.unfavorable_label = 0
        dataset_aif360.favorable_label = 1
    #np.delete(dataset_aif360.features, SENSIBLE_ATT_INDEX, axis=1)    
    sec_ml_dataset = CDataset(dataset_aif360.features , dataset_aif360.labels)
    
    return sec_ml_dataset.X.get_data(), sec_ml_dataset.Y.get_data(), SENSIBLE_ATT_INDEX


def train_LogReg(training_set, test_set):
    
    
    # Metric to use for training and performance evaluation
    # Creation of the multiclass classifier
    metric = CMetricAccuracy()

    #clf = CClassifierSVM(kernel=CKernelRBF()) # Radial Basis Function (RBF) kernel.
    #clf = CClassifierSVM(kernel=CKernelLinear()) # Linear kernel.
    clf = CClassifierLogistic()
    # Parameters for the Cross-Validation procedure
    xval_params = {'C': [1, 10]}#, 'kernel.gamma': [0.1]}#, 5, 10, 25, 50, 100]}

    # Let's create a 3-Fold data splitter
    
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Select and set the best training parameters for the classifier
    print("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=training_set,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )

    print("The best training parameters are: ", best_params)
    # We can now fit the classifier
    clf.fit(training_set)
    print("Training of classifier complete!")

    # Compute predictions on a test set
    y_pred = clf.predict(test_set.X)

    # Evaluate the accuracy of the classifier
    acc = metric.performance_score(y_true=test_set.Y, y_pred=y_pred)

    print("Accuracy on test set: {:.2%}".format(acc))
    
    return clf, acc


def get_error_rates(y_true, y_pred, sensible_att_vals, privileged_classes=1, favorable_output=1, verbose=False):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    

    sensible_att_vals = np.array(sensible_att_vals)
    
    
    
    privileged_y_true = y_true[sensible_att_vals == privileged_classes]
    unprivileged_y_true = y_true[sensible_att_vals != privileged_classes]
    
    privileged_y_pred = y_pred[sensible_att_vals == privileged_classes]
    unprivileged_y_pred = y_pred[sensible_att_vals != privileged_classes]
    """
    privileged_num_errors = len(privileged_y_true) - (len(np.where(np.isclose(privileged_y_true, privileged_y_pred))[0]))
    unprivileged_num_errors = len(unprivileged_y_true) - (len(np.where(np.isclose(unprivileged_y_true, unprivileged_y_pred))[0]))
    
    if verbose:
        print("\tN1: ", n1)
        print("\tN2: ", n2)

        error_rate = (unprivileged_num_errors / len(unprivileged_y_true)) / (privileged_num_errors / len(privileged_y_true))
    """
    
    FNR_privileged = get_false_negative_rate(privileged_y_true, privileged_y_pred, favorable_output)
    FNR_unprivileged = get_false_negative_rate(unprivileged_y_true, unprivileged_y_pred, favorable_output)
    
    FPR_privileged = get_false_positive_rate(privileged_y_true, privileged_y_pred, favorable_output)
    FPR_unprivileged = get_false_positive_rate(unprivileged_y_true, unprivileged_y_pred, favorable_output)
    
    
    if verbose:
        print("\tFNR_1: ", FNR_privileged)
        print("\tFNR_2: ", FNR_unprivileged)
        print("\tFPR_1: ", FPR_privileged)
        print("\tFPR_2: ", FPR_unprivileged)
    
    FNR = -1
    FPR = -1
    
    try:
        FNR = FNR_unprivileged / FNR_privileged
    except:
        pass
    
    try:
        FPR = FPR_unprivileged / FPR_privileged
    except:
        pass
        

    return ({"FNR": FNR, "FNR_privileged":FNR_privileged, "FNR_unprivileged":FNR_unprivileged, "FNR": 1}, {"FPR":FPR, "FPR_privileged":FPR_privileged, "FPR_unprivileged":FPR_unprivileged})


def get_false_positive_rate(y_true, y_pred, favorable_output):
    _tmp1 = y_pred[y_true!=favorable_output]
    fp = _tmp1[_tmp1 == favorable_output]
    
    N = len(y_true[y_true != favorable_output])
    
    if N == 0:
        return 0
    
    return len(fp) / N

def get_true_positive_rate(y_true, y_pred, favorable_output):
    _tmp1 = y_pred[y_true==favorable_output]
    fp = _tmp1[_tmp1 == favorable_output]
    
    P = len(y_true[y_true == favorable_output])
    
    if P == 0:
        return 0
    
    return len(fp) / P
    
    
def get_false_negative_rate(y_true, y_pred, favorable_output):
    _tmp = y_pred[y_true==favorable_output]
    
    fn = _tmp[_tmp != favorable_output]
    
    P = len(y_true[y_true != favorable_output])
    
    if P == 0:
        return 0
    
    return len(fn) / P


def get_average_odds_difference(y_true, y_pred, sensible_att_vals, privileged_classes=1, favorable_output=1):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)    

    sensible_att_vals = np.array(sensible_att_vals)
                                 
    privileged_y_true = y_true[sensible_att_vals == privileged_classes]
    unprivileged_y_true = y_true[sensible_att_vals != privileged_classes]
    
    privileged_y_pred = y_pred[sensible_att_vals == privileged_classes]
    unprivileged_y_pred = y_pred[sensible_att_vals != privileged_classes]
                                 
                                 
    FPR_unprivileged = get_false_positive_rate(unprivileged_y_true, unprivileged_y_pred, favorable_output)
    FPR_privileged = get_false_positive_rate(privileged_y_true, privileged_y_pred, favorable_output)
    TPR_unprivileged = get_true_positive_rate(unprivileged_y_true, unprivileged_y_pred, favorable_output)
    TPR_privileged = get_true_positive_rate(privileged_y_true, privileged_y_pred, favorable_output)
                              
    return 0.5 * abs((FPR_unprivileged - FPR_privileged) + (TPR_unprivileged - TPR_privileged))
    

def train_SVM(training_set, test_set):
    
    
    # Metric to use for training and performance evaluation
    # Creation of the multiclass classifier
    metric = CMetricAccuracy()

    #clf = CClassifierSVM(kernel=CKernelRBF()) # Radial Basis Function (RBF) kernel.
    clf = CClassifierSVM(kernel=CKernelLinear()) # Linear kernel.
    #clf = CClassifierLogistic()
    # Parameters for the Cross-Validation procedure
    xval_params = {'C': [1, 10]}#,'kernel.gamma': [0.1, 5, 10, 25, 50, 100]}

    # Let's create a 3-Fold data splitter
    
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Select and set the best training parameters for the classifier
    print("Estimating the best training parameters...")
    best_params = clf.estimate_parameters(
        dataset=training_set,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        perf_evaluator='xval'
    )

    print("The best training parameters are: ", best_params)

    # We can now fit the classifier
    clf.fit(training_set)
    print("Training of classifier complete!")

    # Compute predictions on a test set
    y_pred = clf.predict(test_set.X)

    # Evaluate the accuracy of the classifier
    acc = metric.performance_score(y_true=test_set.Y, y_pred=y_pred)

    print("Accuracy on test set: {:.2%}".format(acc))
    
    return clf, acc

def execute_normal_poisoning_attack(surrogate_clf, training_set, validation_set, test_set, sensible_att_in_test, privileged_condition_validation, percentage_pois=0.2):
    
    metric = CMetricAccuracy()
    NUM_SAMPLES_TRAIN = training_set.num_samples
    n_poisoning_points = int(NUM_SAMPLES_TRAIN * percentage_pois) # Number of poisoning points to generate
    print("Creating {} poisoning samples ".format(n_poisoning_points))
    # Should be chosen depending on the optimization problem
    solver_params = {
        'eta': 0.05,
        'eta_min': 0.05,
        'eta_max': None,
        'max_iter': 1000,
        'eps': 1e-6
    }


    pois_attack = CAttackPoisoningLogisticRegression(classifier=surrogate_clf,
                                      training_data=training_set,
                                      surrogate_classifier=surrogate_clf,
                                      surrogate_data=validation_set,
                                      val=validation_set,
                                      distance='l1',
                                      dmax=10,
                                      lb=validation_set.X.min(), ub=validation_set.X.max(),
                                      solver_params=solver_params,
                                      random_seed=random_state,
                                      init_type="random")

    pois_attack.n_points = n_poisoning_points
    
    #dimp_loss = CLossDisparateImpact(privileged_condition_validation)
    #pois_attack._attacker_loss = dimp_loss

    # Run the poisoning attack
    print("Attack started...")
    pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(test_set.X, test_set.Y)
    print("Attack complete!")
       
    pois_tr = training_set.deepcopy().append(pois_ds)  # Join the training set with the poisoning points
   
    return pois_ds, pois_tr
    
    