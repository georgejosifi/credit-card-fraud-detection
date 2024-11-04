import dacite, yaml
import shutil
import pandas as pd
from common.dataclasses import PipelineConfig, LoadedData
from pathlib import Path
from common.util import load_data, create_pipeline, get_hyperparameter_tuning_algorithm_and_params, get_imbalanced_learn_algorithm
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from datetime import datetime

SAVED_RESULTS_DIR = Path('saved_results/')
PIPELINE_DIR = Path(__file__).parent

def tune_hyperparameters(pipeline: Pipeline, data: LoadedData, pipelineConfig: PipelineConfig):
    hyperparameter_tuning_algo, hyperparameter_tuning_params = get_hyperparameter_tuning_algorithm_and_params(pipelineConfig)
    clf = hyperparameter_tuning_algo(estimator = pipeline, **hyperparameter_tuning_params)

    X_train, X_test, y_train, y_test = train_test_split(data.train_values,data.target_values, train_size= 0.8)

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    recall = recall_score(y_true= y_test, y_pred = y_pred)
    f1 = f1_score(y_true = y_test, y_pred = y_pred)

    print(f'These are the best parameters {clf.best_params_}')
    print(f'The best score: {clf.best_score_}')
    print(f'Unbiased recall score on test set: {recall}')
    print(f'Unbiased f1 score on test set: {f1}')

    print(clf.cv_results_)
    


def validate(pipeline: Pipeline, data: LoadedData, pipeline_config: PipelineConfig):
    skf =  StratifiedKFold(n_splits=5, shuffle= True)
    recall_scores = []
    f1_scores = []
    

    for i, (train_indices, test_indices) in enumerate(skf.split(data.train_values,data.target_values)):
        print(f'Starting Fold: {i}')
        
        X_train, X_test = data.train_values.loc[train_indices], data.train_values.loc[test_indices]
        y_train, y_test = data.target_values.loc[train_indices], data.target_values.loc[test_indices]

        if pipeline_config.imbalanced_learn:
            print('Doing imbalanced sampling')
            sampling_algorithm = get_imbalanced_learn_algorithm(pipeline_config)
            classification_step = pipeline.steps.pop(-1)
            X_train = pipeline.fit_transform(X_train)
            X_train, y_train = sampling_algorithm.fit_resample(X_train, y_train)
            classification_step[1].fit(X_train, y_train)
            pipeline.steps.append(classification_step)
        else:
            pipeline.fit(X_train,y_train)

        
        y_pred = pipeline.predict(X_test)

        recall = recall_score(y_true= y_test, y_pred= y_pred)
        f1 = f1_score(y_true=y_test, y_pred= y_pred)
        recall_scores.append(recall)
        f1_scores.append(f1)

    recall_score_mean = round(np.mean(recall_scores),4)
    recall_std = round(np.std(recall_scores),4)
    f1_score_mean = round(np.mean(f1_scores),4)
    f1_std = round(np.std(f1_scores),4)

    
    print(f'Recall score: {recall_score_mean}, Std: {recall_std}')
    print(f'F1 score: {f1_score_mean}, Std: {f1_std}')

    return {
        'Recall score: ': float(recall_score_mean),
        'Recall Std: ': float(recall_std),
        'F1 score: ': float(f1_score_mean),
        'F1 std: ': float(f1_std)
     }


def predict(pipeline: Pipeline, pipeline_config: PipelineConfig, data: LoadedData):
    pipeline.fit(X=data.train_values,y=data.target_values)
    predictions = pipeline.predict(X=data.test_values)
    predictions= pd.Series(data=predictions)
    
    output_file = Path(pipeline_config.results_dir)/ pipeline_config.output_file
    pd.Series.to_csv(predictions, output_file, index= False)







    

def main():
    config_file = PIPELINE_DIR/ "config" / "config.yaml" 
    with config_file.open('r') as f:
        pipeline_config: PipelineConfig = dacite.from_dict(data_class= PipelineConfig, data = yaml.load(Loader= yaml.FullLoader, stream=f))
    
    loaded_data = load_data(pipeline_config)

    
    pipeline = create_pipeline(pipeline_config)
    if pipeline_config.validate:
        result_scores = validate(pipeline, loaded_data, pipeline_config)

        config_dir = PIPELINE_DIR/SAVED_RESULTS_DIR/datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        config_dir.mkdir(parents= True)
        score_file = config_dir/ 'scores.yaml'
        with score_file.open('w') as f:
            yaml.dump(result_scores,f)
        
        shutil.copytree(PIPELINE_DIR/ pipeline_config.steps_dir, config_dir/'result_steps')
    
    else:
        predict(pipeline,pipeline_config, loaded_data)



if __name__ == "__main__":
    main()

