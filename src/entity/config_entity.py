import os 
from dataclasses import dataclass
from src.constant.training_pipeline import *
from datetime import datetime
from src.constant.prediction_pipeline import PRED_SCHEMA_FILE_PATH
TIMESTAMP: str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

from pymongo import MongoClient
from src.utils.main_utils import MainUtils
from src.constant.s3_bucket import TRAINING_BUCKET_NAME, PREDICTION_BUCKET_NAME
@dataclass
class TrainingPipelineConfig:
    pipeline_name:str =PIPELINE_NAME
    artifact_dir :str=os.path.join(PIPELINE_NAME,ARTIFACT_DIR,TIMESTAMP)
    timestamp:str =TIMESTAMP


# created the object of the class 
training_pipeline_config=TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str =os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)  ## THIS IS THE BASE DIRECTORY
    feature_store_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    ingested_data_dir:str=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR)
    training_file_path:str=os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME



@dataclass
class DataValidationConfig:
    data_validation_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    valid_data_dir:str =os.path.join(data_validation_dir,DATA_VALIDATION_VALID_DIR)
    invalid_data_dir:str=os.path.join(data_validation_dir,DATA_VALIDATION_INVALID_DIR)
    valid_train_file_path:str=os.path.join(valid_data_dir,TRAIN_FILE_NAME)
    valid_test_file_path:str=os.path.join(valid_data_dir,TEST_FILE_NAME)
    invalid_train_file_path: str = os.path.join(invalid_data_dir, TRAIN_FILE_NAME)
    invalid_test_file_path: str = os.path.join(invalid_data_dir, TEST_FILE_NAME)
    drift_report_file_path:str=os.path.join(data_validation_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    data_transformation_dir:str=os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_FILE_NAME.replace("csv","npy"))
    transformed_test_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FILE_NAME.replace("csv","npy"))
    transformed_vectorizer_object_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,VECTORIZER__OBJECT_FILE_NAME)
    transformed_encoded_onject_file_path:str=os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,ENCODER_OBJECT_FILE_NAME)

@dataclass 
class ModelTrainerConfig:
    model_trainer_dir:str= os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
    trained_model_path:str= os.path.join(model_trainer_dir,MODEL_TRAINER_DIR_NAME,MODEL_FILE_NAME)
    excepted_accuracy:float=MODEL_TRAINER_EXPECTED_SCORE
    model_config_file_path:str=MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    



@dataclass
class ModelEvaluationConfig:

    changed_threshold_score:float=MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    model_evaluation_dir:str=os.path.join(training_pipeline_config.artifact_dir,MODEL_EVALUATION_DIR_NAME)
    best_model_path:str=os.path.join(model_evaluation_dir,MODEL_FILE_NAME)


# @dataclass
# class ModelPusherConfig:
#     bucket_name: str = MODEL_PUSHER_BUCKET_NAME
#     s3_model_key_path: str = MODEL_FILE_NAME


class PCAConfig:
    def __init__(self):
        self.n_components = 2
        self.random_state = 42
        
    def get_pca_config(self):
        return self.__dict__
    
class ClusteringConfig:
    def __init__(self):
        self.n_clusters=3
        self.affinity='euclidean'
        self.linkage='ward'
    
    def get_clustering_config(self):
        return self.__dict__




class SimpleImputerConfig:
    def __init__(self):
        self.strategy = "constant"

        self.fill_value = 0



class Prediction_config:
    def __init__(self):
        utils = MainUtils()
        self.prediction_schema = utils.read_yaml_file(PRED_SCHEMA_FILE_PATH)
    def get_prediction_schema(self):
        return self.__dict__
    


