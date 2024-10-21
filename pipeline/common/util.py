from pathlib import Path
import yaml, dacite
from common.dataclasses import PipelineConfig, LoadedData, ClassifierConfig, TransformerConfig
from common.transformers import get_transformer_algorithm
from common.classifiers import get_classification_algorithm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

PIPELINE_DIR = Path(__file__).parent.parent

def load_data(pipeline_Config: PipelineConfig) -> LoadedData:
    data_file = PIPELINE_DIR / pipeline_Config.data_dir/ pipeline_Config.data_file
    data = pd.read_csv(data_file)
    loaded_data = LoadedData(train_values= data.drop([pipeline_Config.target_name], axis = 1), 
                             target_values= data[pipeline_Config.target_name])

    return loaded_data


def create_pipeline(pipeline_Config: PipelineConfig) -> Pipeline:
    steps_dir_path = PIPELINE_DIR / pipeline_Config.steps_dir
    sorted_steps_dir_path = sorted(steps_dir_path.iterdir())
    classifier_config: ClassifierConfig
    transformer_configs = []
    for i, step in enumerate(sorted_steps_dir_path):
        with step.open('r') as f:
            config  = yaml.load(f,yaml.FullLoader)

        config['step_name'] = step.name.split('_')[1].split('.')[0]

        if config['step_name'] == 'classification':
            classifier_config = dacite.from_dict(data_class= ClassifierConfig, data = config)
            print('Loaded classifier')

        else:
            config: TransformerConfig = dacite.from_dict(data_class= TransformerConfig, data = config)
            transformer_configs.append(config)
            print(f"Loaded step {i+1}: {config.step_name}") 

    transformers = []
    for transformer_conf in transformer_configs:

        transformer = get_transformer_algorithm(transformer_conf.transformer_type, transformer_conf.transformer_algorithm, transformer_conf.algorithm_parameters)
        transformers.append((transformer_conf.name, transformer, transformer_conf.features))
    
    column_transformer  = ColumnTransformer(transformers= transformers, verbose_feature_names_out= False, remainder = 'passthrough')
    column_transformer.set_output(transform= 'pandas')
    
    classifier = get_classification_algorithm(classifier_config.classification_algorithm,classifier_config.algorithm_parameters)

    pipeline = Pipeline(steps= [('preprocessing',column_transformer), ('classification',classifier)], verbose= True)
    return pipeline




    


        



    







