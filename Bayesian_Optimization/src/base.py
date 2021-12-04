from bayes_opt import BayesianOptimization, UtilityFunction

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
        
        # if optimizer == 'vanilla':
        #     self.optimizer = BayesianOptimization(  f = evaluation,
        #                                             pbounds=pbounds,
        #                                             verbose=2,
        #                                             random_state=random_state,)
        # else:
        #     raise NotImplementedError
    
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

    

    
