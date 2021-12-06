from bayes_opt import BayesianOptimization, UtilityFunction

""" 
    Base class of Bayesian Optimization

Other BO can be heritaged from this base.
    
    base functions:
    
    next_to_probe(): the suggest data of BO
    
    evaluate(): evaluation metric based on self.evaluation
    
"""
class BaseBO:
    def __init__(self,acq_function,evaluation,) -> None:
        # if acq_function['name'] == "ucb":
        #     kappa = acq_function['kappa']
        #     xi = acq_function['xi']
        #     self.acq_function = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)
        # else:
        #     raise NotImplementedError
        
        self.acq_function = acq_function
        
        self.evaluation = evaluation
        
    
    def next_to_probe(self,):
        return self.optimizer.suggest(self.acq_function)

    def evaluate(self,model,dataset,deo,**x):
        return self.evaluation(model,dataset,deo,**x)

    # def iterate(self,T,budget):

    #     for _ in range(budget):
    #         next_point = self.next_to_probe()
    #         self.optimizer.register(
    #                             params=next_point,
    #                             target=self.evaluate(**next_point),
    #                         )

    

    
