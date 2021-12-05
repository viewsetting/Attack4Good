# Bayesian Optimization

This is the folder of bayesian optimization based poison attack algorithm.

## Requirements

```
numpy, scipy, seaborn, matplotlib, scikit-learn, pandas, bayesian_optimization
```

## Files

```
.
├── README.md   // README file
├── run_gb.sh	// shell script of gradient boosting
├── run_rf.sh	// shell script of random forest
└── src 
    ├── base.py		// base class of Bayesian Optimization (BaseBO)
    ├── classifier	// sklearn classfifiers
    │   ├── GB.py	
    │   ├── LR.py	
    │   ├── RF.py	
    ├── main.py		// main functions (random_seed_test(), run_compas())
    ├── utils		// utility for BO
    │   ├── data_preprocessing.py	// class of Compas
    │   └── fairness_metric.py		// several metrics for fairness
    └── vanilla_BO.py		// implementation of vanilla BO (VanillaBO) and function of iterative attack: run_vanillabo()
```



## How to run it

Modify shell files to run experiments.

### Use python command 

Excute the command below in terminal.

- model: [ 'rf ', 'gb' ], 'rf' for random forest and 'gb' for gradient boosting
- budget: integer, the running time of BO
- add_label: the label of poison data point

```
python src/main.py --model gb --budget 495 --add_label '1'
```



### random forest

```bash
sh run_rf.sh
```

### gradient boost

```bash
sh run_gb.sh
```

