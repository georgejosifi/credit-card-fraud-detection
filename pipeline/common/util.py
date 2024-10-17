from pathlib import Path
from common.dataclasses import PipelineConfig, LoadedData
import pandas as pd
from sklearn.pipeline import Pipeline


def load_data(pipeline_Config: PipelineConfig) -> LoadedData:
    pipeline_dir = Path(__file__).parent.parent
    train_values_file = pipeline_dir / pipeline_Config.data_dir/ pipeline_Config.data_file
    train_values_data = pd.read_csv(train_values_file)
    loaded_data = LoadedData(train_values=train_values_data)

    return loaded_data

