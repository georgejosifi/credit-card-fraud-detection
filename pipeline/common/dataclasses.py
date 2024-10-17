from dataclasses import dataclass
import pandas as pd

@dataclass
class PipelineConfig:
    data_dir: str
    data_file: str
    target_name: str


@dataclass
class LoadedData:
    train_values: pd.DataFrame


