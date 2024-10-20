from enum import Enum
from typing import Any

from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.preprocessing import (Binarizer, KBinsDiscretizer, MaxAbsScaler,
                                   MinMaxScaler, Normalizer, OneHotEncoder,
                                   OrdinalEncoder, QuantileTransformer,
                                   RobustScaler, StandardScaler)
from sklearn.random_projection import GaussianRandomProjection

import category_encoders as ce

from custom_transformers.remove_columns_transformer import ZeroTransformer



class ImputationAlgorithm(str, Enum):
    SimpleImputer = "SimpleImputer"
    KNNImputer = "KNNImputer"
    IterativeImputer = "IterativeImputer"


imputation_algorithms = {
    ImputationAlgorithm.SimpleImputer: SimpleImputer,
    ImputationAlgorithm.KNNImputer: KNNImputer,
    ImputationAlgorithm.IterativeImputer: IterativeImputer
}


class PreprocessingAlgorithm(str, Enum):
    Binarizer = "Binarizer"
    KBinsDiscretizer = "KBinsDiscretizer"
    MaxAbsScaler = "MaxAbsScaler"
    MinMaxScaler = "MinMaxScaler"
    OneHotEncoder = "OneHotEncoder"
    OrdinalEncoder = "OrdinalEncoder"
    QuantileTransformer = "QuantileTransformer"
    RobustScaler = "RobustScaler"
    StandardScaler = "StandardScaler"
    Normalizer = "Normalizer"
    TargetEncoder = "TargetEncoder"
    MEstimateEncoder = "MEstimateEncoder"
    LeaveOneOutEncoder = "LeaveOneOutEncoder"
    GLMMEncoder = "GLMMEncoder"
    HelmertEncoder = "HelmertEncoder"
    JamesSteinEncoder = "JamesSteinEncoder"
    CatBoostEncoder = "CatBoostEncoder"
    ZeroTransformer = "ZeroTransformer"


preprocessing_algorithms = {
    PreprocessingAlgorithm.Binarizer: Binarizer,
    PreprocessingAlgorithm.KBinsDiscretizer: KBinsDiscretizer,
    PreprocessingAlgorithm.MaxAbsScaler: MaxAbsScaler,
    PreprocessingAlgorithm.MinMaxScaler: MinMaxScaler,
    PreprocessingAlgorithm.OneHotEncoder: OneHotEncoder,
    PreprocessingAlgorithm.OrdinalEncoder: OrdinalEncoder,
    PreprocessingAlgorithm.QuantileTransformer: QuantileTransformer,
    PreprocessingAlgorithm.RobustScaler: RobustScaler,
    PreprocessingAlgorithm.StandardScaler: StandardScaler,
    PreprocessingAlgorithm.Normalizer: Normalizer,
    PreprocessingAlgorithm.TargetEncoder: ce.TargetEncoder,
    PreprocessingAlgorithm.MEstimateEncoder: ce.MEstimateEncoder,
    PreprocessingAlgorithm.LeaveOneOutEncoder: ce.LeaveOneOutEncoder,
    PreprocessingAlgorithm.GLMMEncoder: ce.GLMMEncoder,
    PreprocessingAlgorithm.HelmertEncoder: ce.HelmertEncoder,
    PreprocessingAlgorithm.JamesSteinEncoder: ce.JamesSteinEncoder,
    PreprocessingAlgorithm.CatBoostEncoder: ce.CatBoostEncoder,
    PreprocessingAlgorithm.ZeroTransformer: ZeroTransformer
}


class DimensionalityReductionAlgorithm(str, Enum):
    PCA = "PCA"
    FeatureAgglomeration = "FeatureAgglomeration"
    GaussianRandomProjection = "GaussianRandomProjection"


dimensionality_reduction_algorithms = {
    DimensionalityReductionAlgorithm.PCA: PCA,
    DimensionalityReductionAlgorithm.FeatureAgglomeration: FeatureAgglomeration,
    DimensionalityReductionAlgorithm.GaussianRandomProjection: GaussianRandomProjection
}

class TransformerType(str, Enum):
    Imputation = "Imputation"
    Preprocessing = "Preprocessing"
    DimensionalityReduction = "DimensionalityReduction"


transformer_algorithms = {
    TransformerType.Imputation: (ImputationAlgorithm, imputation_algorithms),
    TransformerType.Preprocessing: (PreprocessingAlgorithm, preprocessing_algorithms),
    TransformerType.DimensionalityReduction: (
        DimensionalityReductionAlgorithm, dimensionality_reduction_algorithms),
}

def get_transformer_algorithm(transformer_type: str, transformer_name: str, parameters: dict[str,Any]):
    try:
        algorithms_type, algorithm_dict = transformer_algorithms[TransformerType[transformer_type]]
    except:
        print(f'Error: Only the following transformer types are available {[e.value for e in TransformerType]} but {transformer_type} was given')
        exit()
    try:
        transformer_algorithm = algorithm_dict[algorithms_type[transformer_name]]

    except:
        print(f'Error: From {transformer_type} only the following algorithms are available {[e.value for e in algorithms_type]} but {transformer_name} was given')

    instance = transformer_algorithm(**parameters)

    return instance



