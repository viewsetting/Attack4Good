import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

class Compas:
    """[class of COMPAS dataset]
    """
    def __init__(self, shuffle=True,
                       test_size=0.2,
                       valid_size=0.1,
                       random_state=1,
                       valid = True) -> None:

        self.dataset = fetch_openml('compas-two-years', as_frame=True, )
        self.X = self.dataset.data
        self.Y = self.dataset.target
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.dataset.data,self.dataset.target, 
                                                                                test_size=test_size,
                                                                                shuffle=shuffle,
                                                                                random_state=random_state,
                                                                                )
        self.p_bound = {
                        'sex':(0,1),
                        'age':(18,80),
                        'juv_fel_count' : (0,10),
                        'juv_misd_count':(0,10),
                        'juv_other_count':(0,10),
                        'priors_count':(0,25), 
                        'age_cat_25_45':(0,1),
                        'age_cat_Greaterthan45':(0,1),
                        'age_cat_Lessthan25':(0,1),
                        'race_African_American':(0,1),
                        'race_Caucasian':(0,1),
                        'c_charge_degree_F':(0,1),
                        'c_charge_degree_M':(0,1)   } 
        
        if valid is True:
            self.X_train, self.X_valid, self.Y_train, self.Y_valid = train_test_split(self.dataset.data,self.dataset.target,
                                                                                test_size=valid_size, 
                                                                                shuffle=(shuffle^3)%1000007,
                                                                                random_state=random_state,
                                                                                )
    def get_train(self,is_numpy=True):
        if is_numpy:
            return self.X_train.to_numpy(), self.Y_train.to_numpy()
        else:
            return self.X_train, self.Y_train
    def get_valid(self,is_numpy=True):
        if is_numpy:
            return self.X_valid.to_numpy(), self.Y_valid.to_numpy()
        else:
            return self.X_valid, self.Y_valid
    def get_test(self,is_numpy=True):
        if is_numpy:
            return self.X_test.to_numpy(), self.Y_test.to_numpy()
        else:
            return self.X_test, self.Y_test

    """to_numpy_x_hat
    Convert from numerical type input type to original type
    """
    def to_numpy_x_hat(self,x_dict,race_thresold=0.3):
        x = np.zeros((13,))
        x[0] = str(round((x_dict['sex'])))
        x[1] = float(round(x_dict['age']))
        x[2] = float(round(x_dict['juv_fel_count']))
        x[3] = float(round(x_dict['juv_misd_count']))
        x[4] = float(round(x_dict['juv_other_count']))
        x[5] = float(round(x_dict['priors_count']))

        # x[6] = 1.0 if x_dict['age_cat_25_45'] > x_dict['age_cat_Greaterthan45'] and x_dict['age_cat_25_45'] > x_dict['age_cat_Lessthan25'] else 0.0
        # x[7] = 1.0 if x_dict['age_cat_Greaterthan45'] > x_dict['age_cat_25_45'] and x_dict['age_cat_Greaterthan45'] > x_dict['age_cat_Lessthan25'] else 0.0
        # x[8] = 1.0 if x_dict['age_cat_Lessthan25'] > x_dict['age_cat_25_45'] and x_dict['age_cat_Lessthan25'] >x_dict['age_cat_Greaterthan45'] else 0.0
        
        x[6] = 0.0
        x[7] = 0.0
        x[8] = 0.0
        
        if x[0] < 25.0:
            x[8] = 1.0
        elif x[0] < 45:
            x[6] = 1.0
        else:
            x[7] = 1.0

        x[9] = '1' if x_dict['race_African_American'] > x_dict['race_Caucasian'] and x_dict['race_African_American'] > race_thresold else '0'
        x[10] = '0' if x_dict['race_Caucasian'] > x_dict['race_African_American'] and x_dict['race_Caucasian'] > race_thresold else '0'

        x[11] = float(round(x_dict['c_charge_degree_F']))
        x[12] = float(round(x_dict['c_charge_degree_M']))

        return x
    
    
    

    
    def add_data(self,max_param,label='1'):
        """[add poisoned data]

        Args:
            max_param ([type]): [max_param of BO]
            label (str, optional): [description]. Defaults to '1'.
        """
        print('Poinson data ADDED')
        print('feature: ',max_param.keys())
        np_max_param = self.to_numpy_x_hat(max_param)
        print('val: ',np_max_param)
        self.X_train = np.row_stack((self.X_train,np_max_param))
        self.Y_train = np.append(self.Y_train,label)                                                                   


if __name__ == "__main__":    
    c = Compas()

    x,y = c.get_train(is_numpy=False)
    x_t,y_t = c.get_test()
    print(x_t[:10])
    

    from sklearn.linear_model import LogisticRegression

    m = LogisticRegression()
    m.fit(x,y)

    res = m.predict(x_t)
    print(res)
    
    c.add_data(x_t[1])
    xx,_ = c.get_train()
    print(xx[-1],xx[1])

    
    