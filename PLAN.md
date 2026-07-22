To Build a production Grade ML Model pipeline which is served as FastAPI Containerized APP on Cloud with Git and Automated Github Testing 



Flow:
1. Dataset: 
    - Clean the Dataset in Notebook , Prepare for ML Model training

2. Training:
    - train the Model with Experimental Tracing/Record Keeping using MLFLOW 
    - Evaluation with proper metrics

3. Backend:
    - Wrap the Entire thing in FastApi backend 
    - Containerize it using Docker 

4. Tests and Scripts : 
    - Add tess of : Data Validation of User Inputs, APIs, Training , etc. using Pytest
    - Add scripts of training, config etc.


5. Github WorkFlow
    - Add Automated script to Test in yml files


6. Deploy to Cloud using simple way 




Dataset and Model Versioning :

"""
Test: all test executed in ci or local saves the model and dataset in local folder structure Issue..
backend : takes the model from local file strucutre , need to take from Model/Dataset Versioning 

Thus Dataset and model version need to be external source, otherwise the folder strucutre get congested with models/dataset.

Dataset and Model need to be versioned in WandB registry.. ?
- thus server loads the models (classifier, preprocessor) from wandb whichever will be production env 
- tests execution :
    CI:
        load three type of dataset and model from wandb -> execute tests -> save artifacts back in wandb as per user collections [here no saving of artifacts locally]
    Local:
        change tests to take local model/dataset if needed -> execute tests -> if need tracing save in wandb

What to change:
- initial dataset in registry 
- scripts to save in Registry of wandb both models and dataset or link to dataset
- tests passes from where to load dataset and models to test 
- dev changes this when pushing to remote to load from registry

"""
