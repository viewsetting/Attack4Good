
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


custom_palette=sns.diverging_palette(255, 133, l=60, n=12, center="dark")
sns.palplot(sns.color_palette("Paired", 12))
sns.set_palette(sns.color_palette("Paired"))
sns.set_style("white")
sns.set_context("paper", rc={"font.size":16,"axes.titlesize":20,"axes.labelsize":16})

import math
import numpy as np
import matplotlib.pyplot as plt # for plotting stuff
from random import seed, shuffle
from scipy.stats import multivariate_normal # generating synthetic data
SEED = 999
seed(SEED) # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)

def get_data():
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
    dataset_orig_train, dataset_orig_vt = dataset_aif360.split([0.2], shuffle=True)
    dataset_orig_valid, dataset_orig_test = dataset_orig_vt.split([0.5], shuffle=True)
    
    dataset_aif360 = dataset_orig_train.copy()
    
    SENSIBLE_ATT_INDEX = dataset_orig.feature_names.index(dataset_orig.protected_attribute_names[0])

    # print('The unfavorable label is ' , dataset_aif360.unfavorable_label )
    ## Correcting labels assignation
    if dataset_aif360.unfavorable_label != 0:
        Y = dataset_aif360.labels
        Y[Y == dataset_aif360.unfavorable_label] = -1
        Y[Y == dataset_aif360.favorable_label] = 1
        Y[Y == -1] = 0

        dataset_aif360.unfavorable_label = 0
        dataset_aif360.favorable_label = 1
    #np.delete(dataset_aif360.features, SENSIBLE_ATT_INDEX, axis=1)    
    sec_ml_dataset = CDataset(dataset_aif360.features , dataset_aif360.labels)
    
    return sec_ml_dataset.X.get_data(), sec_ml_dataset.Y.get_data(), SENSIBLE_ATT_INDEX
    
x = get_data()

# print(a,b,c)
# p_priv = (y == 0).sum() / y.size
# p_unpriv = (y == 1).sum() / y.size
# print(p_priv, p_unpriv)


from sklearn.model_selection import train_test_split
from utils_Our import calculate_disparate_impact, get_data2, train_LogReg, get_error_rates, get_average_odds_difference, train_SVM, execute_normal_poisoning_attack
from attack_Our import execute_adversarial_attack

N = 9
n =1 
dimp_in_data = []
euc_distances = []
dimp_scenarios = []
## Generating data
X,y,sensitive_att_index = get_data()
formatted_X=X ## Concatenating X with sensible att

print("X_shape: ", X.shape)
sensible_att_all = X[:,sensitive_att_index]  # The sensitive attention of X, in CAMPUS, X[0]
sec_ml_dataset_all = CDataset(X, y)
# print(sec_ml_dataset_all)

dimp_in_data.append(calculate_disparate_impact(sec_ml_dataset_all.Y.get_data(), sensible_att_all)) 

## Splitting data. 
X_train_val, X_test, y_train_val, y_test = train_test_split(formatted_X, y, test_size=0.2, random_state=random_state)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.5, random_state=random_state)

# print("X_train: ", X_train.shape)
training = CDataset(X_train, y_train)
training_sensible_att = X_train[:,sensitive_att_index]

validation = CDataset(X_val, y_val)
validation_sensible_att = X_val[:,sensitive_att_index]
val_lambda = np.zeros(validation.num_samples)

## Creating lambda vector
val_lambda[np.where((validation_sensible_att==0) & (y_val==0))[0]] == 1 ## Unprivileged denied
val_lambda[np.where((validation_sensible_att==0) & (y_val==1))[0]] == 1 ## Unprivileged granted
val_lambda[np.where((validation_sensible_att==1) & (y_val==0))[0]] == -1 ## Privileged denied
val_lambda[np.where((validation_sensible_att==1) & (y_val==1))[0]] == -1 ## Privileged granted

# print(val_lambda.shape)
test = CDataset(X_test, y_test)
test_sensible_att = X_test[:,sensitive_att_index]



## GENERATING DATA FOR WHITE BOX ATTACK
X2,y2,sensitive_att_index2 = get_data2()
formatted_X2=X2 ## Concatenating X with sensible att

sec_ml_dataset_all2 = CDataset(X2, y2)
sensible_att_all2 = sec_ml_dataset_all2.X.get_data()[:,sensitive_att_index2]

## Splitting data. 
X_train_val2, X_test2, y_train_val2, y_test2 = train_test_split(formatted_X2, y2, test_size=0.2, random_state=random_state)
X_train2, X_val2, y_train2, y_val2 = train_test_split(X_train_val2, y_train_val2, test_size=0.5, random_state=random_state)

print('Posion data size', X2.shape) 

training2 = CDataset(X_train2, y_train2)
training_sensible_att2 = X_train2.ravel()

validation2 = CDataset(X_val2, y_val2)
validation_sensible_att2 = X_val2.ravel()
val_lambda2 = np.zeros(validation2.num_samples)

test2 = CDataset(X_test2, y_test2)
test_sensible_att2 = X_test2.ravel()

scenario = {
    "name": "Use case 4 - {}".format(n),
    "description": "Disparate impact attack. \n Euclidean distance between group averages: {}\n".format(n),
    "training": training,
    "training_sensible_att" : training_sensible_att,
    "validation" : validation,
    "validation_sensible_att" : validation_sensible_att,
    "lambda_validation" : validation_sensible_att,
    "test": test,
    "test_sensible_att" : test_sensible_att,
    "all_data" : sec_ml_dataset_all,
    "all_sensible_att" : sensible_att_all,        
    "black_box_training": training2,
    "black_box_training_sensible_att" : training_sensible_att2,
    "black_box_validation" : validation2,
    "black_box_validation_sensible_att" : validation_sensible_att2,
    "black_box_lambda_validation" : val_lambda2,
    "black_box_test": test2,
    "black_box_test_sensible_att" : test_sensible_att2,
    "black_box_all_data" : sec_ml_dataset_all2,
    "black_box_all_sensible_att" : sensible_att_all2,
}
    
    
dimp_scenarios.append(scenario)



for scenario in dimp_scenarios:
    print("\n\n ==== {} ====".format(scenario['name']))
    print("    - {}\n".format(scenario['description']))
    
    ################################
    ### ORIGINAL CLF PERFORMANCE ###
    ################################
    original_model, original_acc = train_LogReg(scenario["training"], scenario["test"])
    
    orig_y_pred = original_model.predict(scenario["test"].X)
    orig_FNR, orig_FPR = get_error_rates(scenario["test"].Y.get_data(), orig_y_pred.get_data(), scenario["test_sensible_att"], 1, 1)
    orig_disparate_imp = calculate_disparate_impact(orig_y_pred.get_data(), scenario["test_sensible_att"])
    orig_odds_diff = get_average_odds_difference(scenario["test"].Y.get_data(), orig_y_pred.get_data(), scenario["test_sensible_att"])

    scenario['original_classifier'] = original_model
    scenario['original_acc'] = original_acc
    scenario['orig_d_imp'] = orig_disparate_imp
    scenario['orig_FNR'] = orig_FNR
    scenario['orig_FPR'] = orig_FPR
    scenario['orig_odds'] = orig_odds_diff
    

    ########################
    ### WHITE BOX ATTACK ###
    ########################
    white_pois_clf = deepcopy(original_model)
    
    privileged_condition_valid = np.ones(scenario['validation'].num_samples)
    privileged_condition_valid[scenario["validation_sensible_att"] == 0] == -1
    
    
    white_pois_points, white_pois_tr = execute_adversarial_attack(white_pois_clf, scenario["training"], scenario["validation"], scenario["test"], scenario["test_sensible_att"], scenario["validation_sensible_att"])
    ## Retraining with poisoned points
    white_pois_clf = white_pois_clf.fit(white_pois_tr)
    white_pois_y_pred = white_pois_clf.predict(scenario["test"].X)
    
    metric = CMetricAccuracy()
    white_pois_acc = metric.performance_score(scenario["test"].Y, y_pred=white_pois_y_pred)
    white_pois_disparate_imp = calculate_disparate_impact(white_pois_y_pred.get_data(), scenario["test_sensible_att"])
    white_pois_FNR, white_pois_FPR = get_error_rates(scenario["test"].Y.get_data(), white_pois_y_pred.get_data(), scenario["test_sensible_att"], 1, 1)
    white_odds_diff = get_average_odds_difference(scenario["test"].Y.get_data(), white_pois_y_pred.get_data(), scenario["test_sensible_att"])

    scenario['white_poisoned_classifier'] = white_pois_clf
    scenario['white_poisoned_points'] = white_pois_points
    scenario['white_pois_d_imp'] = white_pois_disparate_imp
    scenario['white_pois_y_pred'] = white_pois_y_pred
    scenario['white_pois_acc'] = white_pois_acc
    scenario['white_pois_FNR'] = white_pois_FNR
    scenario['white_pois_FPR'] = white_pois_FPR
    scenario['white_odds'] = white_odds_diff
    
    
    
    ########################
    ### BLACK BOX ATTACK ###
    ########################
    real_model, real_acc = train_SVM(scenario["training"], scenario["test"])
    
    surrogate_clf = deepcopy(original_model)
    
    black_pois_points, black_pois_tr = execute_adversarial_attack(surrogate_clf, scenario["training"], scenario["validation"], scenario["test"], scenario["test_sensible_att"], scenario["validation_sensible_att"])
    ## Retraining with poisoned points
    
    black_pois_clf = deepcopy(real_model)
    black_pois_clf = black_pois_clf.fit(black_pois_tr)
    black_pois_y_pred = black_pois_clf.predict(scenario["test"].X)
    
    black_pois_acc = metric.performance_score(y_true=scenario["test"].Y, y_pred=black_pois_y_pred)
    black_pois_disparate_imp = calculate_disparate_impact(black_pois_y_pred.get_data(), scenario["test_sensible_att"])
    black_pois_FNR, black_pois_FPR = get_error_rates(scenario["test"].Y.get_data(), black_pois_y_pred.get_data(), scenario["test_sensible_att"], 1, 1)
    black_odds_diff = get_average_odds_difference(scenario["test"].Y.get_data(), black_pois_y_pred.get_data(), scenario["test_sensible_att"])

    scenario['black_poisoned_classifier'] = black_pois_clf
    scenario['black_poisoned_points'] = black_pois_points
    scenario['black_pois_d_imp'] = black_pois_disparate_imp
    scenario['black_pois_y_pred'] = black_pois_y_pred
    scenario['black_pois_acc'] = black_pois_acc
    scenario['black_pois_FNR'] = black_pois_FNR
    scenario['black_pois_FPR'] = black_pois_FPR
    scenario['black_odds'] = black_odds_diff
    
    
    
    ################################
    ### CLASSIC POISONING ATTACK ###
    ################################
    normal_pois_clf = deepcopy(original_model)
    
    privileged_condition_valid = np.ones(scenario['validation'].num_samples)
    privileged_condition_valid[scenario["validation_sensible_att"] == 0] == -1
    
    
    normal_pois_points, normal_pois_tr = execute_normal_poisoning_attack(normal_pois_clf, scenario["training"], scenario["validation"], scenario["test"], scenario["test_sensible_att"], scenario["validation_sensible_att"])
    ## Retraining with poisoned points
    normal_pois_clf = normal_pois_clf.fit(normal_pois_tr)
    normal_pois_y_pred = normal_pois_clf.predict(scenario["test"].X)
    
    metric = CMetricAccuracy()
    normal_pois_acc = metric.performance_score(scenario["test"].Y, y_pred=normal_pois_y_pred)
    print("->> normal")
    normal_pois_disparate_imp = calculate_disparate_impact(normal_pois_y_pred.get_data(), scenario["test_sensible_att"])
    normal_odds_diff = get_average_odds_difference(scenario["test"].Y.get_data(), normal_pois_y_pred.get_data(), scenario["test_sensible_att"])
    normal_pois_FNR, normal_pois_FPR = get_error_rates(scenario["test"].Y.get_data(), normal_pois_y_pred.get_data(), scenario["test_sensible_att"], 1, 1)
    


    scenario['normal_poisoned_classifier'] = normal_pois_clf
    scenario['normal_poisoned_points'] = normal_pois_points
    scenario['normal_pois_d_imp'] = normal_pois_disparate_imp
    scenario['normal_odds'] = normal_odds_diff
    scenario['normal_pois_y_pred'] = normal_pois_y_pred
    scenario['normal_pois_acc'] = normal_pois_acc
    scenario['normal_pois_FNR'] = normal_pois_FNR
    scenario['normal_pois_FPR'] = normal_pois_FPR
    



from matplotlib.gridspec import GridSpec
def plot_disparate_impact(scenarios, data_dimp=None , n_scenarios=1, title=None):
    
    x1 = [n for n in range(n_scenarios)]
    
    x1 = 1
    x2 = 2.5
    x3 = 4
    
    fig=plt.figure(figsize=[12,8])
    if title is not None:
        plt.suptitle(title, fontsize=14)

    gs=GridSpec(2,4) # 2 rows, 3 columns

    ax1=fig.add_subplot(gs[0,:2]) # Second row, span all columns
    ax1.set_title("Accuracy")
    
    ax1.bar(x1,[float(UC["original_acc"]) for UC in scenarios], color="darkgreen", label="Original classifier.")
    ax1.bar(x2,[float(UC["white_pois_acc"]) for UC in scenarios], color="red", label="Poisoned classifier. White box attack.")
    ax1.bar(x3,[float(UC["black_pois_acc"]) for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack.")
    

    ax2=fig.add_subplot(gs[0,2]) # First row, first column
    ax2.set_title("DP")
    ax2.bar(x1,[UC["orig_d_imp"] for UC in scenarios], color="darkgreen", label="Original classifier.")
    #if data_dimp is not None:
    #ax2.bar(0,data_dimp, color="darkgrey", label="Original data")
    ax2.bar(x2,[UC["white_pois_d_imp"] for UC in scenarios], color="red", label="Poisoned classifier. White box attack.")
    ax2.bar(x3,[UC["black_pois_d_imp"] for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack.")

    ax2b=fig.add_subplot(gs[0,3]) # First row, second column
    ax2b.set_title("AOD")
    ax2b.bar(x1, [UC["orig_odds"] for UC in scenarios], color="darkgreen", label="Original classifier.")
    ax2b.bar(x2, [UC["white_odds"] for UC in scenarios], color="red", label="Poisoned classifier.White box attack")
    ax2b.bar(x3, [UC["black_odds"] for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack")

    
    ax3=fig.add_subplot(gs[1,0]) # First row, second column
    ax3.set_title("FNR privileged")
    ax3.bar(x1,[UC["orig_FNR"]["FNR_privileged"] for UC in scenarios], color="darkgreen", label="Original classifier.")
    ax3.bar(x2,[UC["white_pois_FNR"]['FNR_privileged'] for UC in scenarios], color="red", label="Poisoned classifier. White box attack.")
    ax3.bar(x3,[UC["black_pois_FNR"]['FNR_privileged'] for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack.")

    ax4=fig.add_subplot(gs[1,1]) # First row, third column
    ax4.set_title("FNR unprivileged")
    ax4.bar(x1,[UC["orig_FNR"]["FNR_unprivileged"] for UC in scenarios], color="darkgreen", label="Original classifier.")
    ax4.bar(x2,[UC["white_pois_FNR"]['FNR_unprivileged'] for UC in scenarios], color="red", label="Poisoned classifier. White box attack.")
    ax4.bar(x3,[UC["black_pois_FNR"]['FNR_unprivileged'] for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack.")
    #ax4.bar([], [], color="gray", label="Disparate impact in the data")
    ax4.legend(bbox_to_anchor=(1.8, -0.1),fontsize=15)
    
    ax5=fig.add_subplot(gs[1,2]) # First row, second column
    ax5.set_title("FPR privileged")
    ax5.bar(x1,[UC["orig_FPR"]["FPR_privileged"] for UC in scenarios], color="darkgreen", label="Original classifier.")
    ax5.bar(x2,[UC["white_pois_FPR"]['FPR_privileged'] for UC in scenarios], color="red", label="Poisoned classifier. White box attack.")
    ax5.bar(x3,[UC["black_pois_FPR"]['FPR_privileged'] for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack.")

    ax6=fig.add_subplot(gs[1,3]) # First row, third column
    ax6.set_title("FPR unprivileged")
    ax6.bar(x1,[UC["orig_FPR"]["FPR_unprivileged"] for UC in scenarios], color="darkgreen", label="Original classifier.")
    ax6.bar(x2,[UC["white_pois_FPR"]['FPR_unprivileged'] for UC in scenarios], color="red", label="Poisoned classifier. White box attack.")
    ax6.bar(x3,[UC["black_pois_FPR"]['FPR_unprivileged'] for UC in scenarios], color="orange", label="Poisoned classifier. Black box attack.")
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_xticks([])
    ax4.set_xticks([])
    ax5.set_xticks([])
    ax6.set_xticks([])
    
    
    plt.plot()
    plt.figure(figsize=[4,2])
    plt.savefig('/home/gubin/boyangsun/Poisoning-Attacks-on-Algorithmic-Fairness-master/SecML/fairness/Output2.png')

plot_disparate_impact(dimp_scenarios, dimp_in_data)
