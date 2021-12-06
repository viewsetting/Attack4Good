from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from scipy.sparse import data
from base import BaseBO
from utils.fairness_metric import deo

def get_utility_function(kind="ucb"):
    """[get utility function]

    Args:
        kind (str, optional): [utility function]. Defaults to "ucb".

    Returns:
        [UtilityFunction]: [UtilityFunction]
    """
    if kind == "ucb":
        return UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
    else:
        return NotImplementedError

def get_vanilla_BO(utility_function,evaluation,p_bound):
    """[get VanillaBO]

    Args:
        utility_function ([UtilityFunction]): [UtilityFunction]
        evaluation ([function]): [function of evaluation]
        p_bound ([dict]): [p_bound]

    Returns:
        [VanillaBO]: [VanillaBO]
    """
    return VanillaBO(
    utility_function,
    evaluation,
    pbounds=p_bound,
    random_state=1,
)

class VanillaBO(BaseBO):
    """[VanillaBO]

    Args:
        BaseBO ([BaseBO]): [base of Bayesian Optimization]
    """
    def __init__(self,acq_function,evaluation,pbounds,random_state=1):
        BaseBO.__init__(self,acq_function=acq_function,evaluation=evaluation)
        self.optimizer = BayesianOptimization(  f = evaluation,
                                                    pbounds=pbounds,
                                                    verbose=2,
                                                    random_state=random_state,)

def run_vanillabo(model,dataset,bb_function,add_label,pre_target,budget):
    """[run vanillabo]

    Args:
        model ([sklearn model]): [model of scikit-learn]
        dataset ([dataset]): [class of dataset]
        bb_function ([function]): [blackbox evaluation]
        add_label (str, optional): [the label of poison data to add]
        pre_target ([float]): [previous target of fairness metric]
        budget (int, optional): [the budget of BO]. 

    Returns:
        [dataset,cnt,target]: [poisoned dataset, budget that has been used in inner loop, new best fairness metric value]
    """

    utility = get_utility_function(kind="ucb")
    cnt = 0
    bo = get_vanilla_BO(utility,bb_function,p_bound=dataset.p_bound)
    
    # if cnt == 1:
        
    #     next_point_to_probe = bo.next_to_probe()
        
    #     target = bo.evaluate(**next_point_to_probe)
        
    #     bo.optimizer.register(params=next_point_to_probe,target=target)
        
    #     return model.score(dataset.X_val,dataset.Y_val.reshape(-1,1))
        
    #     pass
    
    #x_train, y_train = dataset.get_train()
    x_val, y_val = dataset.get_valid(True)
    
    while(cnt<=budget):
        cnt+=1
        next_point_to_probe = bo.next_to_probe()
        
        target = bo.evaluate(model,dataset,deo,**next_point_to_probe)
        
        bo.optimizer.register(params=next_point_to_probe,target=target)
        
        if pre_target < target:
            print('PRETARGET: {}, TARGET: {}'.format(-pre_target,-target))
            print('ACC: {}'.format(model.score(x_val,y_val.reshape(-1,1))))
            
            dataset.add_data(bo.optimizer.max['params'],label=add_label)
            
            pre_target = target
            
            return dataset,cnt,target
        
    print('The end.')
    return dataset, -1, -1
    
    
    

if __name__ == '__main__':
    run_vanillabo()
