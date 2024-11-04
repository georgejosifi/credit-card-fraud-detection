# How to run

1. Put the data in the ```data``` folder.
2. Run pipeline.py

You will have the prediction results in the ```results``` folder.

# How to configure the pipeline

1. Put the steps in the ```config/steps``` folder.
2. The last step has to have the name ```classification```.
3. Use ```config/config.yaml``` to choose whether you want to validate a model, get predictions for a test file, or tune hyperparameters
3. For hyperparameter tuning use ```config/hyperparameter-tuning.yaml```
4. Try sampling algorithms using ```config/imbalanced_learning.yaml```
