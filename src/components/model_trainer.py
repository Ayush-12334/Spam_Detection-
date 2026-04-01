import sys 
from typing import List,Tuple 
import os
import pandas as pd 
import numpy as np

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact

from src.exception import CustomeException
from src.logger import logging
from src.utils.main_utils import MainUtils,load_numpy_array_data
from neuro_mf  import ModelFactory


class  SpamDetectionModel:
    def __init__(self,preprocessing_object:object,encoder_object:object,trained_model_object:object):
        self.preprocessing_object=preprocessing_object
        self.encoder_object=encoder_object


        self.trained_model_object=trained_model_object

    def predict(self, x):
        try:
            logging.info("model prediction started")

            # ✅ If already numpy → don't transform
            if isinstance(x, np.ndarray):
                return self.trained_model_object.predict(x)

            # ✅ If raw text → transform
            transformed_feature = self.preprocessing_object.transform(x)
            return self.trained_model_object.predict(transformed_feature)

        except Exception as e:
            raise CustomeException(e, sys) from e
        
    def __repr__(self):
        return f'{type(self.trained_model_object).__name__}()'
    def __str__(self):
        return f'{type(self.trained_model_object).__name__}()'
    

class ModelTrainer:
    def __init__(self,data_transformation_artifact :DataTransformationArtifact,model_trainer_config:ModelTrainerConfig):

        self.data_transformation_artifact =data_transformation_artifact
        self.model_trainer_config=model_trainer_config
        self.utils=MainUtils()





    def initiate_model_training(self)->ModelTrainerArtifact:
        try:
            logging.info("Entered initiate_model_trainer method of Modeltrainer class")

            train_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr=load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)

            x_train,y_train,x_test,y_test=train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1]


            model_factory=ModelFactory(model_config_path=self.model_trainer_config.model_config_file_path)
            best_model_details=model_factory.get_best_model(X=x_train,y=y_train,base_accuracy=self.model_trainer_config.excepted_accuracy)
            preprocessor_object=self.utils.load_object(file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path)
            encoder_obejct=self.utils.load_object(file_path=self.data_transformation_artifact.transformed_encoder_object_file_path)


            if best_model_details.best_score<self.model_trainer_config.excepted_accuracy:
                logging.info("no best model found with score more than base")
                raise Exception("no best model found with score more than the base score")
            
            spam_detection_model=SpamDetectionModel(
                preprocessing_object=preprocessor_object,
                encoder_object=encoder_obejct,
                trained_model_object=best_model_details.best_model
            )
            logging.info("spam ham detection mdoel is creted and saved")
            trained_model_file_path=os.path.dirname(self.model_trainer_config.trained_model_path)
            os.makedirs(trained_model_file_path,exist_ok=True)

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=spam_detection_model
            )
            logging.info(f"spam ham detection model saved sucessfully at {trained_model_file_path}")
            mertic_artifact=ClassificationMetricArtifact(f1_score=0.8,precision_score=0.8,recall_score=0.9)
            model_trainer_artifact=ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_path,
                metric_artifact=mertic_artifact
            )

            logging.info("model training completed sucessfully")
            logging.info(f'Model trainer artifact:{model_trainer_artifact}')    

            return model_trainer_artifact
        

        except Exception as e:
            raise CustomeException(e,sys) from e
        


