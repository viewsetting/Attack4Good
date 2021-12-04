#SEC_ML imports
from secml.data.c_dataset import CDataset
from secml.ml.classifiers import CClassifierSVM, CClassifierLogistic
from secml.ml.kernels import CKernelRBF, CKernelLinear
from secml.ml.peval.metrics import CMetricAccuracy
from secml.data.splitter import CDataSplitterKFold

# Poisoning attacks
from secml.adv.attacks import CAttackPoisoningSVM
from secml.adv.attacks.poisoning.c_attack_poisoning_logistic_regression import CAttackPoisoningLogisticRegression
random_state = 999
from secml.ml.classifiers.loss import CLossClassification, CLossLogistic
from secml.array import CArray

def execute_adversarial_attack(surrogate_clf, training_set, validation_set, test_set, sensible_att_in_test, privileged_condition_validation, percentage_pois=0.2):
    
    print(" ==> Adversarial attack. Percentage of samples: {} ".format(percentage_pois))
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
                                      distance='l2',
                                      dmax=10,
                                      lb=validation_set.X.min(), ub=validation_set.X.max(),
                                      solver_params=solver_params,
                                      random_seed=random_state,
                                      init_type="random")

    pois_attack.n_points = n_poisoning_points
    
    dimp_loss = CLossDisparateImpact(privileged_condition_validation)
    pois_attack._attacker_loss = dimp_loss

    # Run the poisoning attack
    print("Attack started...")
    pois_y_pred, pois_scores, pois_ds, f_opt = pois_attack.run(test_set.X, test_set.Y)
    print("Attack complete!")

    
    pois_tr = training_set.deepcopy().append(pois_ds)  # Join the training set with the poisoning points
   
    return pois_ds, pois_tr




class CLossDisparateImpact(CLossClassification):
    """Surrogate function of disparate impact.

    Attributes
    ----------
    class_type : 'log'
    suitable_for : 'classification'

    """
    __class_type = 'dimp_log'

    def __init__(self, _privileged_condition):
        self._privileged_condition = CArray(_privileged_condition)

    def unprivileged(self):
        """Give 1 to unprivileged, 0 to privileged"""
        y = CArray.zeros(self._privileged_condition.size)
        y[self._privileged_condition == 0] = 1
        return y

    def loss(self, y_true, score, pos_label=1):
        """Computes loss_priv-loss_unpriv, which is what we aim to max"""
        # give 1 to unpriv, 0 to priv
        y = self.unprivileged()
        p_priv = (y == 0).sum() / y.size
        p_unpriv = (y == 1).sum() / y.size
        # loss = (score >= 0) != y  # zero-one loss
        loss = CLossLogistic().loss(y_true=y, score=score)  # smoothed version
        loss[y == 1] *= p_priv / p_unpriv  # rebalance class weights
        # loss[y == 1] *= -p_unpriv / p_priv
        return -abs(loss)

    def dloss(self, y_true, score, pos_label=1):
        """Computes the derivative of the loss vs score."""
        y = self.unprivileged()
        p_priv = (y == 0).sum() / y.size
        p_unpriv = (y == 1).sum() / y.size
        grad = CLossLogistic().dloss(y, score, pos_label)
        grad[y == 1] *= p_priv / p_unpriv  # rebalance class weights
        # grad[y == 1] *= -p_unpriv / p_priv
        return grad