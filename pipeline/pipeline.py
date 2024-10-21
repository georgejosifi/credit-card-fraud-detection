import dacite, yaml
from common.dataclasses import PipelineConfig, LoadedData
from pathlib import Path
from common.util import load_data, create_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np



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

    
    print(f'Recall score: {round(np.mean(recall_scores),4)}, Std: {round(np.std(recall_scores),4)}')
    print(f'F1 score: {round(np.mean(f1_scores),4)}, Std: {round(np.std(f1_scores),4)}')


    

def main():
    pipeline_dir = Path(__file__).parent
    config_file = pipeline_dir/ "config" / "config.yaml" 
    with config_file.open('r') as f:
        pipeline_config: PipelineConfig = dacite.from_dict(data_class= PipelineConfig, data = yaml.load(Loader= yaml.FullLoader, stream=f))
    
    loaded_data = load_data(pipeline_config)

    
    pipeline = create_pipeline(pipeline_config)

    validate(pipeline,loaded_data)



    


if __name__ == "__main__":
    main()

