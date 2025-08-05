"""
Tags are essential in our use case (especially architecture and fold) bacasuse they allow us to filter neccessary runs to extract the best hyperparameter combination from folds.
Because we got folds we need to get averages per hyperparameter combination across folds (for each architecture).
So, from num_architectures * num_folds * num_combination we should get num_architectures of models.
We will filter by tags, averege per combination and pick the highest value for each architecture.
"""
import matplotlib.pyplot as plt
from mlflow.pytorch import load_model
from mlflow.tracking import MlflowClient

# TODO: implement getting averages per combination of hyperparameters across all folds
def avg_combinations(models, metrics):
    pass