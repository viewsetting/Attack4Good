# Bayesian Optimization

This is the folder of bayesian optimization based poison attack algorithm.

## Requirements

```
numpy, scipy, seaborn, matplotlib, scikit-learn, pandas, bayesian_optimization
```

## How to run it

Modify shell files to run experiments.

### Use python command 

Excute the command below in terminal.

- Model: [ 'rf ', 'gb' ]
- Budget: integer, the running time of BO
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

