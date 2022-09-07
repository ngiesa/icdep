

## This GitHub repository includes the following files:

|py file|description|
|--|--|
| preprocessing_layer.preprocessor | class for preprocessing data focusing on converting raw data |
| preprocessing_layer.time_manager | class for handling time infos per surgery and managing time windows T1-T3 |
| preprocessing_layer.annotator | class for annoatation of target variable based on NuDesc assessment |
| preprocessing_layer.test_train_splitter | class for initially splitting train and test on basis of patient ids |
| preprocessing_layer.data_manager | class for accessing, transformation and analyzing preprocessed data |
| preprocessing_layer.data_standardizer | class for performing z transformation on train and test splits |
| statistic_layer.univariate_statistic | class performing MWU and OR for measuring effect size per feature |
| statistic_layer.feature_selector | class defining, storing and selection of feature sets |
| modelling_layer.model_trainer | class configurating experiments for hyperparameter search via cv |
| modelling_layer.train_functions | collection of functions used from class in model_trainer |
| modelling_layer.hypertuner | class for configuration and execution of hyperparameter tuning |
| modelling_layer.metrics | collection of functions calcualting and preparing performance metrics |
| modelling_layer.model_evaluator | class for performing bootstrapped evaluations with performance metrics |
| modelling_layer.comparison_models | class for retraining and evaluating of baseline models|


## Using stored pretrained models

The pretrained models for predicting POD are stored under modeling_layer.models.saved_models as pickle files. In order to load them into your project, you need to run "pip install pickle" when using Python < 3.9. Otherwise you can just run "import pickle" in your script where you want to use these models. Models are labeled as "sav_model_M{T}" where {T} represents a placeholder for the corresponding perioperative time phases:

|time phase| description |
|--|--|
|T1 | preoperative time |
|T2 | intraoperative time |
|T3 | postoperative time |
|T12 | T1 + T2 |
|T23 | T2 + T3 |
|T123 | T1 + T2 + T3 |

You may want to preprocess your features that are used by the stored models first. Please read the feature encoding and preprocessing part of the paper included in Multimedia Appendix 2. You find a list of all input features per model variant in the X column of the file  modeling_layer.models.saved_models.models_metadata.csv. Please make sure that one row in your data structure for the feature space corresponds to one surgery (instead of patient or case) and columns represent features. Make also sure to keep the order of features described in the models_metadata.csv file. 

Afterwards you can run "model = pickle.load(open('./modeling_layer/models/saved_models/sav_model_M{T}.pkl', 'rb'))" to load your model. When having X_test holding the preapred feature space you make predictions with "model.predict(X_test)".
