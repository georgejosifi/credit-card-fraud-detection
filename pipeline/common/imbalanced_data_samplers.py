from enum import Enum

from imblearn.over_sampling import (RandomOverSampler, SMOTE, SMOTEN, SMOTENC, ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, ClusterCentroids, CondensedNearestNeighbour, EditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, TomekLinks)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier


class OversamplingAlgorithm(str, Enum):
    RandomOverSampler = "RandomOverSampler"
    SMOTE = "SMOTE"
    SMOTEN = "SMOTEN"
    SMOTENC = "SMOTENC"
    ADASYN = "ADASYN"
    BorderlineSMOTE = "BorderlineSMOTE"
    KMeansSMOTE = "KMeansSMOTE"
    SVMSMOTE = "SVMSMOTE"

oversampling_algorithms = {
    OversamplingAlgorithm.RandomOverSampler: RandomOverSampler,
    OversamplingAlgorithm.SMOTE: SMOTE,
    OversamplingAlgorithm.SMOTEN: SMOTEN,
    OversamplingAlgorithm.SMOTENC: SMOTENC,
    OversamplingAlgorithm.ADASYN: ADASYN,
    OversamplingAlgorithm.BorderlineSMOTE: BorderlineSMOTE,
    OversamplingAlgorithm.KMeansSMOTE: KMeansSMOTE,
    OversamplingAlgorithm.SVMSMOTE: SVMSMOTE
}
  
class UndersamplingAlgorithm(str, Enum):
    RandomUnderSampler = "RandomUnderSampler"
    ClusterCentroids = "ClusterCentroids"
    CondensedNearestNeighbour = "CondensedNearestNeighbour"
    EditedNearestNeighbours = "EditedNearestNeighbours"
    AllKNN = "AllKNN"
    InstanceHardnessThreshold = "InstanceHardnessThreshold"
    NearMiss = "NearMiss"
    NeighbourhoodCleaningRule = "NeigbourhoodCleaningRule"
    OneSidedSelection = "OneSidedSelection"
    TomekLinks = "TomekLinks"
    
undersampling_algorithms = {
    UndersamplingAlgorithm.RandomUnderSampler: RandomUnderSampler,
    UndersamplingAlgorithm.ClusterCentroids: ClusterCentroids,
    UndersamplingAlgorithm.CondensedNearestNeighbour: CondensedNearestNeighbour,
    UndersamplingAlgorithm.EditedNearestNeighbours: EditedNearestNeighbours,
    UndersamplingAlgorithm.AllKNN: AllKNN,
    UndersamplingAlgorithm.InstanceHardnessThreshold: InstanceHardnessThreshold,
    UndersamplingAlgorithm.NearMiss: NearMiss,
    UndersamplingAlgorithm.NeighbourhoodCleaningRule: NeighbourhoodCleaningRule,
    UndersamplingAlgorithm.OneSidedSelection: OneSidedSelection,
    UndersamplingAlgorithm.TomekLinks: TomekLinks
}
  
class OverAndUndersamplingAlgorithm(str, Enum):
    SMOTEENN = "SMOTEENN",
    SMOTETomek = "SMOTETomek"

over_and_undersampling_algorithms = {
    OverAndUndersamplingAlgorithm.SMOTEENN: SMOTEENN,
    OverAndUndersamplingAlgorithm.SMOTETomek: SMOTETomek
}

class SamplerType(str, Enum):
    OversamplingAlgorithm = "OversamplingAlgorithm"
    UndersamplingAlgorithm = "UndersamplingAlgorithm"
    OverAndUndersamplingAlgorithm = "OverAndUndersamplingAlgorithm"

samplers = {
    SamplerType.OversamplingAlgorithm: (OversamplingAlgorithm, oversampling_algorithms),
    SamplerType.UndersamplingAlgorithm: (UndersamplingAlgorithm, undersampling_algorithms),
    SamplerType.OverAndUndersamplingAlgorithm: (OverAndUndersamplingAlgorithm, over_and_undersampling_algorithms),
}

def get_sampling_algorithm(sampler_name: str, algorithm_name: str, algorithm_parameters):
    try:
        algorithm_types, sampler_dict = samplers[SamplerType[sampler_name]]
    except:
        print(f'Only the following sampler types are supported {[e.value for e in SamplerType]} but {sampler_name} was given')
        exit()

    try:
        algorithm = sampler_dict[algorithm_types[algorithm_name]]
    except:
        print(f'From {algorithm_types} only the following algorithms {[e.value for e in algorithm_types]} are available but {algorithm_name} was given')
        exit()

    instance = algorithm(**algorithm_parameters)
    return instance