from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from enum import Enum


class ParameterTuningAlgorithm(str,Enum):
    GridSearchCV = "GridSearchCV"
    RandomizedSearchCV = "RandomizedSearchCV"


algorithms = {
    ParameterTuningAlgorithm.GridSearchCV: GridSearchCV,
    ParameterTuningAlgorithm.RandomizedSearchCV: RandomizedSearchCV
}


def get_hyperparameter_tuning_algorithm(algorithm_name: str):
    try:
        algorithm = algorithms[ParameterTuningAlgorithm[algorithm_name]]
    except:
        print(f'The available hyper parameter tuning algorithms are: {[e.value for e in ParameterTuningAlgorithm]} but {algorithm_name} was given')
        exit()

    return algorithm


