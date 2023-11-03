# Description and Argumentations
## Argumentation about model selection.
> Argumentation about selection of xgboost as model. It has better accurancy 
keeping in mind the possibility of scale the features of the model,
since for now the target it is delay maybe in the future there is other multidimensional
approaches that can be handled better in performance by xgboost.

## Argumentation about deploy it into AWS.
> Since it is based docker container it is better to use an approach based on kubernetes cluster to deploy it.
That will helps to improve the scalability, decoupling and orchestration of the deployment and execution, scaling containers
on demand about the resources usage.

The infrastructure architecture looks like this:





![diagram](https://github.com/dsotoduque/predict-app/assets/17690605/7be2e0a7-77b3-4759-aed9-91ee6071a892)
