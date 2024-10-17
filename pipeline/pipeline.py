import dacite, yaml
from common.dataclasses import PipelineConfig, LoadedData
from pathlib import Path
from common.util import load_data



def main():
    pipeline_dir = Path(__file__).parent
    config_file = pipeline_dir/ "config" / "config.yaml" 
    with config_file.open('r') as f:
        pipeline_config: PipelineConfig = dacite.from_dict(data_class= PipelineConfig, data = yaml.load(Loader= yaml.FullLoader, stream=f))
    
    loaded_data = load_data(pipeline_config)

    print(loaded_data.train_values)


    


if __name__ == "__main__":
    main()

