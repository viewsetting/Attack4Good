# main.py
from operator import add
import numpy as np
from vanilla_BO import run_vanillabo
from utils.data_preprocessing import Compas
from classifier.GB import get_GradientBoostingClassifier
from classifier.RF import get_RandomForestClassifier
from utils.fairness_metric import deo
import argparse


def run_compas(model,dataset,budget=500,add_label='1'):
    """[run compas dataset]

    Args:
        model ([sklearn model]): [model of scikit-learn]
        dataset ([dataset(compas)]): [class of COMPAS]
        budget (int, optional): [the budget of BO]. Defaults to 500.
        add_label (str, optional): [the label of poison data to add]. Defaults to '1'.
    """
    def black_box_function(model,dataset,deo,x_hat):
        """[inner function of BO]

        Args:
            model ([sklearn model]): [model of scikit-learn]
            dataset ([dataset(compas)]): [class of COMPAS]
            deo ([function]): [metric of fairness]
            x_hat ([numpy.ndarray]): [array of poinson data]

        Returns:
            [float]: [score of fairness metric]
        """
        #dataset.add_data(x_hat,label=add_label)
        x_train,y_train = dataset.get_train(False)
        x_val,y_val = dataset.get_valid(True)
        x_train = np.row_stack((x_train,x_hat))
        y_train = np.append(y_train,add_label)
        model.fit(x_train,y_train)

        y = model.predict(x_val)
        
        deo_val = deo(y,y_val,x_val)

        #aug
        #deo_val = deo_aug(y,y_val,x_val)
        return deo_val
    
    def discrete_input(model,dataset,deo,sex,age,juv_fel_count,juv_misd_count,juv_other_count,
                    priors_count, age_cat_25_45,age_cat_Greaterthan45,
                    age_cat_Lessthan25,race_African_American,race_Caucasian,c_charge_degree_F,c_charge_degree_M):
        """[transform some continuous variables into categorical]

        Args:
            model ([sklearn model]): [model of scikit-learn]
            dataset ([dataset(compas)]): [class of COMPAS]
            deo ([function]): [metric of fairness]
            sex ([float]): [sex, 0 for male, 1 for female]
            age ([float]): [age]
            juv_fel_count ([float]): [count of felonies]
            juv_misd_count ([float]): [juv_misd_count]
            juv_other_count ([float]): [juv_other_count]
            priors_count ([float]): [total crimes commited before]
            age_cat_25_45 ([float]): [if age beween 25 and 45]
            age_cat_Greaterthan45 ([float]): [if age more than 45]
            age_cat_Lessthan25 ([float]): [if age less than 25]
            race_African_American ([float]): [if African American]
            race_Caucasian ([float]): [if Caucasian]
            c_charge_degree_F ([float]): [c_charge_degree_F]
            c_charge_degree_M ([float]): [c_charge_degree_M]

        Returns:
            [function]: [blackbox evaluate function]
        """
        sex = str(round(sex))
        age = float(round(age))
        juv_fel_count = float(round(juv_fel_count))
        juv_misd_count = float(round(juv_misd_count))
        juv_other_count = float(round(juv_other_count))
        priors_count = float(round(priors_count))

        #race_African_American = str(round(race_African_American))
        #race_Caucasian = str(round(race_Caucasian))
        

        race_African_American = 1 if race_African_American > race_Caucasian and race_African_American > 0.3 else 0
        race_Caucasian = 1 if race_Caucasian > race_African_American and race_Caucasian > 0.3 else 0

        race_African_American=str(race_African_American)
        race_Caucasian = str(race_Caucasian)

        c_charge_degree_F = float(round(c_charge_degree_F))
        c_charge_degree_M = float(round(c_charge_degree_M))


        # age_cat_25_45 = str('1' if (int(age_cat_25_45)> int(age_cat_Greaterthan45) and int(age_cat_25_45) > int(age_cat_Lessthan25) ) else '0')
        # age_cat_Greaterthan45 = str('1' if ( int(age_cat_Greaterthan45) > int(age_cat_25_45) and int(age_cat_Greaterthan45) > int(age_cat_Lessthan25) ) else '0')
        # age_cat_Lessthan25 = str('1' if ( int(age_cat_Lessthan25) > int(age_cat_25_45) and int(age_cat_Lessthan25) > int(age_cat_Greaterthan45) ) else '0')

        age_cat_25_45 = '0'
        age_cat_Greaterthan45='0'
        age_cat_Lessthan25='0'

        if age < 25.0:
            age_cat_Lessthan25 = '1'
        elif age<45.0:
            age_cat_25_45 = '1'
        else:
            age_cat_Greaterthan45 = '1'

        x_hat = np.array([sex,age,juv_fel_count,juv_misd_count,juv_other_count,priors_count,
        age_cat_25_45,age_cat_Greaterthan45,age_cat_Lessthan25,race_African_American,race_Caucasian,
        c_charge_degree_F,c_charge_degree_M])

        return black_box_function(model,dataset,deo,x_hat)      
    
    compas = Compas()
    x_train,y_train = dataset.get_train(True)
    x_val,y_val = dataset.get_valid(True)
    model.fit(x_train,y_train)
    y = model.predict(x_val)
    target = deo(y,y_val,x_val)
    
    
    #inner loop
    while(budget>=0):
        compas, run_time,target = run_vanillabo(model,dataset=compas,bb_function=discrete_input,add_label=add_label,pre_target=target, budget=budget,)
        
        if run_time == -1:
            break
        budget -= run_time
        print("run time: {}".format(run_time))
        print("BUDGET REMAIN: {}".format(budget))
        print('-'*100)
    
    pass

def random_seed_test(kind,budget=495,add_label='1'):
    """[run by setting a list of seeds as initialization]

    Args:
        kind ([str]): ['gb': gradient boost, 'rf': random forest]
        budget (int, optional): [total budget of BO]. Defaults to 495.
        add_label (str, optional): [the label of poison data to add]. Defaults to '1'.

    Raises:
        NotImplementedError: [if the model is not included]
    """
    random_seed = [(x^3-2*x+1)%10007 for x in range(1000,1000+50)]
    
    for seed in random_seed:
        if kind =='gb':
            model = get_GradientBoostingClassifier(random_state=seed)
        elif kind =='rf':
            model = get_RandomForestClassifier(random_state=seed)
        else:
            raise NotImplementedError
        print('*'*50, ' {} '.format(kind) , '*'*50)
        print('SEED: {}'.format(seed))
        compas = Compas()
        run_compas(model,compas,budget=budget,add_label=add_label)

if __name__ == '__main__':
    # model = get_GradientBoostingClassifier()
    # compas = Compas()
    # run_compas(model,compas,budget=495,add_label='1')
    #random_seed_test('gb')
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--add_label", type=str, required=False,default='1', help='1 for criminal, 0 for non-criminal')
    parser.add_argument("--model", type=str,required=True,
                        help='gb for Gradient Boosting classifier , rf for random forest')
    parser.add_argument("--budget", type=int, required=False,default=495, help='Running budget')
    args = parser.parse_args()
    
    add_label = args.add_label
    budget = args.budget
    model = args.model
    
    random_seed_test(kind=model,budget=budget,add_label=add_label)
    
