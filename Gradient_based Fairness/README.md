# Gradient_based Fairness (Appendix)
This code relies on the code from the paper for "Poisoning Attacks on Algorithmic Fairness", ECML 2020. 
One can find out the link in our reference 
(https://arxiv.org/abs/2004.07401)


## Introduction
We followed the method of the paper above which implemented fairness attack through gradient-based update way. By changing the optimization objective and constraint, we aimed at improving the fairness of the trained classifier. 
Code is based on a fork of [SecML](https://secml.github.io/), adapted with new target functions allowing optimize the attacks against Algorithmic Fairness. 

## How to use and extend

The main code is contained in the folder [*SecML/fairness*](https://github.com/dsolanno/Poisoning-Attacks-on-Algorithmic-Fairness/tree/master/SecML/fairness) where there are three main sources:

* First, there is a Python scripts called Ourdemo.py to implement the attack towards fairness on dataset COMPAS.

* Second, there are output images showing the result of attack.


To add new Algorithmic Fairness functions to target, new losses can be added to the [loss](https://github.com/dsolanno/Poisoning-Attacks-on-Algorithmic-Fairness/tree/master/SecML/src/secml/ml/classifiers/loss) folder. Also, experiments with other datasets are welcome!
