from pathlib import Path
import yaml, dacite
from common.dataclasses import PipelineConfig, LoadedData, ClassifierConfig, TransformerConfig, HyperparameterTuningConfig
from common.dataclasses import ImbalancedLearnConfig
from common.transformers import get_transformer_algorithm
from common.classifiers import get_classification_algorithm
from common.hyperparameter import get_hyperparameter_tuning_algorithm
from common.imbalanced_data_samplers import get_sampling_algorithm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from scipy.stats import randint

PIPELINE_DIR = Path(__file__).parent.parent


def load_data(pipeline_Config: PipelineConfig) -> LoadedData:
    data_file = PIPELINE_DIR / pipeline_Config.data_dir/ pipeline_Config.data_file
    test_values_file = PIPELINE_DIR/ pipeline_Config.data_dir/ pipeline_Config.test_file
    data = pd.read_csv(data_file)
    test_values = pd.read_csv(test_values_file)
    loaded_data = LoadedData(train_values= data.drop([pipeline_Config.target_name], axis = 1), 
                             target_values= data[pipeline_Config.target_name], test_values= test_values)

    return loaded_data



def get_imbalanced_learn_algorithm(pipeline_Config: PipelineConfig):
    imbalanced_learn_config_file = PIPELINE_DIR/ pipeline_Config.imbalanced_learning_config_dir
    with imbalanced_learn_config_file.open('r') as f:
        imbalanced_learn_config = dacite.from_dict(data_class=ImbalancedLearnConfig, data=yaml.load(f, yaml.FullLoader))

    sampling_algorithm  =  get_sampling_algorithm(imbalanced_learn_config.sampler_name, 
                                                  imbalanced_learn_config.algorithm_name, 
                                                  imbalanced_learn_config.algorithm_parameters)
    
    return sampling_algorithm
    




def get_hyperparameter_tuning_algorithm_and_params(pipeline_Config: PipelineConfig):
    hyperparameter_tuning_config_file = PIPELINE_DIR/ pipeline_Config.hyperparameter_tuning_config_dir
    with hyperparameter_tuning_config_file.open('r') as f:
        hyperparameter_tuning_config: HyperparameterTuningConfig = dacite.from_dict(data_class= HyperparameterTuningConfig, 
                                                                                    data= yaml.load(f, yaml.FullLoader))
    hyperparameter_tuning_algo =  get_hyperparameter_tuning_algorithm(hyperparameter_tuning_config.algorithm_name)

    algorithm_param = _replace_interval_pattern(hyperparameter_tuning_config.algorithm_parameters)

    return (hyperparameter_tuning_algo, algorithm_param)



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

    column_transformers_list = []
    for transformer_conf in transformer_configs:

        transformer = get_transformer_algorithm(transformer_conf.transformer_type,
                                                 transformer_conf.transformer_algorithm, 
                                                 transformer_conf.algorithm_parameters)
        column_regex = '|'.join(transformer_conf.features)
        
        column_transformer = ColumnTransformer(transformers = [(transformer_conf.name, transformer, make_column_selector(pattern=column_regex))],
                                               verbose_feature_names_out= False,
                                                 remainder = 'passthrough')
        column_transformer.set_output(transform='pandas')
        column_transformers_list.append((transformer_conf.step_name, column_transformer))

    
    classifier = get_classification_algorithm(classifier_config.classification_algorithm,classifier_config.algorithm_parameters)

    pipeline = Pipeline(steps= [*column_transformers_list, ('classification',classifier)], verbose= True)
    return pipeline




def _replace_interval_pattern(hyperparam: dict):
    for key, value in hyperparam.items():
        if isinstance(value,dict):
            _replace_interval_pattern(value)

        if isinstance(value,str):
            if '-' in value:
                interval_endpoints = value.split('-')
                if len(interval_endpoints)>2:
                    hyperparam[key] = [item for item in range(int(interval_endpoints[0]),int(interval_endpoints[1]),int(interval_endpoints[2]))]
                else:
                    hyperparam[key] = [item for item in range(int(interval_endpoints[0]),int(interval_endpoints[1]))]

    return hyperparam


        



    







