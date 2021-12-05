# Attack4Good: Poison attack for improving fairness metric

This is the repository of the final project of MBZUAI ML701 Machine Learning.

We generate poisonous examples **such that** the model, trained on the trainset augmented with those poisonous examples, will be more fair. As any kind of poisonous attacks, this can be formulated as the following bi-level optimization problem, where `h` is the variable we want to optimize, `w` is the variable that is optimized by the inner optimization problem, with `w*` being the optimal value for that problem, and `f` and `g` are two functions.

`argmin_h g(w*, h) s. t. w* \in argmin_w f(w, h)`

For optimizing the outer problem, which is what we actually optimize, we use Bayesian Optimization (it is the algorithms that we tuned). For the inner problem, we use a method that is supposed to represent a typical training algorithm used by some third party, so it is not used and uses default hyparameters values. Typically we use a Random Forest and a Gradient Boosting Algorithm, both of which can be chosen as parameter when running the experiments scripts.

The main experiments are in the ``Bayesian_Optimization`` folder, and the scripts necessary to run those experiments are there.

We also experimented with Gradient Based optimization, which experiments are in the ``Gradient_based fairness `` folder.

Finally, some exploratory notebooks are in the ``notebooks folder``.


These are the results we obtain: 

- For the Random Forest classifier used in the training loop:


![scatter_RF](https://user-images.githubusercontent.com/31916524/144737209-65af5722-8334-4976-aada-339627f91810.png)
![RF_fairness.pdf](https://github.com/viewsetting/Attack4Good/files/7655392/RF_fairness.pdf)
![RF_acc.pdf](https://github.com/viewsetting/Attack4Good/files/7655389/RF_acc.pdf)




- For the Gradient Boosting classifier used in the training loop:

![scatter_GB](https://user-images.githubusercontent.com/31916524/144737211-503c3015-ec47-44b3-bf37-3588f099a636.png)
![GB_fairness.pdf](https://github.com/viewsetting/Attack4Good/files/7655386/GB_fairness.pdf)
 ![GB_acc.pdf](https://github.com/viewsetting/Attack4Good/files/7655388/GB_acc.pdf)
