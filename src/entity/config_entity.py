import os 
from dataclasses import dataclass
from src.constant.training_pipeline import *
from datetime import datetime

TIMESTAMP: str=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

 
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



class SimpleImputerConfig:
    def __init__(self):
        self.strategy = "constant"

        self.fill_value = 0


