from dataclasses import dataclass, field
import pandas as pd
from typing import Any, List

@dataclass
class PipelineConfig:
    data_dir: str
    data_file: str
    target_name: str
    steps_dir: str


@dataclass
class LoadedData:
    train_values: pd.DataFrame
    target_values: pd.DataFrame

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



