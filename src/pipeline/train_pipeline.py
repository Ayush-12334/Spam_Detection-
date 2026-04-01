import sys

from src.components.Data_ingestion import DataIngestion
from src.components.Data_transformation import DataTransformation
from src.components.Data_validation import DataValidation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.exception import CustomeException
from src.logger import logging

from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifacts,
    ModelEvaluationArtifact,
    ModelTrainerArtifact
)

from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(self.data_ingestion_config)
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomeException(e, sys)

    def start_data_validation(self, data_ingestion_artifact):
        try:
            data_validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self.data_validation_config
            )
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise CustomeException(e, sys)

    def start_data_transformation(self, data_ingestion_artifact, data_validation_artifact):
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_artifact=data_validation_artifact,
                data_transformation_config=self.data_transformation_config  # ✅ FIXED
            )
            return data_transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomeException(e, sys)

    def start_model_trainer(self, data_transformation_artifact):
        try:
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            return model_trainer.initiate_model_training()  # ✅ FIXED
        except Exception as e:
            raise CustomeException(e, sys)

    def start_model_evaluation(self, model_trainer_artifact, data_transformation_artifact):
        try:
            model_evaluation = ModelEvaluation(
                model_eval_config=self.model_evaluation_config,
                model_trainer_artifact=model_trainer_artifact,
                data_transformation_artifact=data_transformation_artifact
            )
            return model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise CustomeException(e, sys)

    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()

            data_validation_artifact = self.start_data_validation(
                data_ingestion_artifact
            )

            data_transformation_artifact = self.start_data_transformation(
                data_ingestion_artifact,
                data_validation_artifact
            )

            model_trainer_artifact = self.start_model_trainer(
                data_transformation_artifact
            )

            model_evaluation_artifact = self.start_model_evaluation(
                model_trainer_artifact,
                data_transformation_artifact
            )

            if not model_evaluation_artifact.is_model_accepted:
                logging.info("Model not accepted")
                return None

            logging.info("Pipeline completed successfully ✅")

        except Exception as e:
            raise CustomeException(e, sys)
        


if __name__ == "__main__":
    print(" Running pipeline directly...")
    pipeline = TrainPipeline()
    pipeline.run_pipeline()