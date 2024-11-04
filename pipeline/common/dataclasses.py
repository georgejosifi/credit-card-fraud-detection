from dataclasses import dataclass, field
import pandas as pd
from typing import Any, List

@dataclass
class PipelineConfig:
    validate: bool
    imbalanced_learn: bool
    data_dir: str
    results_dir: str
    data_file: str
    test_file: str
    target_name: str
    output_file: str
    steps_dir: str
    hyperparameter_tuning_config_dir: str
    imbalanced_learning_config_dir: str


@dataclass
class LoadedData:
    train_values: pd.DataFrame
    target_values: pd.DataFrame
    test_values: pd.DataFrame

@dataclass
class TransformerConfig:
    step_name: str
    name: str
    features: List[str]
    transformer_type: str
    transformer_algorithm: str
    algorithm_parameters: dict[str,Any] = field(default_factory= dict)


@dataclass
class ClassifierConfig:
    classification_algorithm: str
    algorithm_parameters: dict[str,Any] = field(default_factory= dict)

@dataclass
class HyperparameterTuningConfig:
    algorithm_name: str
    algorithm_parameters: dict[str,Any]

@dataclass
class ImbalancedLearnConfig:
    sampler_name: str
    algorithm_name: str
    algorithm_parameters: dict[str,Any] = field(default_factory= dict)




