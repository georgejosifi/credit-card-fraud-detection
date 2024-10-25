import dacite, yaml
import shutil
from common.dataclasses import PipelineConfig, LoadedData
from pathlib import Path
from common.util import load_data, create_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from datetime import datetime

SAVED_RESULTS_DIR = Path('saved_results/')
PIPELINE_DIR = Path(__file__).parent



def validate(pipeline, data: LoadedData):
    skf =  StratifiedKFold(n_splits=5, shuffle= True)
    recall_scores = []
    f1_scores = []
    

    for i, (train_indices, test_indices) in enumerate(skf.split(data.train_values,data.target_values)):
        print(f'Starting Fold: {i}')
        
        X_train, X_test = data.train_values.loc[train_indices], data.train_values.loc[test_indices]
        y_train, y_test = data.target_values.loc[train_indices], data.target_values.loc[test_indices]

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



    

def main():
    config_file = PIPELINE_DIR/ "config" / "config.yaml" 
    with config_file.open('r') as f:
        pipeline_config: PipelineConfig = dacite.from_dict(data_class= PipelineConfig, data = yaml.load(Loader= yaml.FullLoader, stream=f))
    
    loaded_data = load_data(pipeline_config)

    
    pipeline = create_pipeline(pipeline_config)

    result_scores = validate(pipeline,loaded_data)

    config_dir = SAVED_RESULTS_DIR/datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    config_dir.mkdir(parents= True)
    score_file = config_dir/ 'scores.yaml'
    with score_file.open('w') as f:
        yaml.dump(result_scores,f)
    
    shutil.copytree(PIPELINE_DIR/ pipeline_config.steps_dir, config_dir/'result_steps')



if __name__ == "__main__":
    main()

