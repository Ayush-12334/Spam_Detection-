import sys
import os
import pandas as pd
import numpy as np

from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
    ClassificationMetricArtifact
)

from src.exception import CustomeException
from src.logger import logging
from src.utils.main_utils import MainUtils, load_numpy_array_data

from neuro_mf import ModelFactory


# =========================
# Model Wrapper
# =========================
class SpamDetectionModel:
    def __init__(self, preprocessing_object: object, encoder_object: object, trained_model_object: object):
        self.preprocessing_object = preprocessing_object
        self.encoder_object = encoder_object
        self.trained_model_object = trained_model_object

    def predict(self, x: pd.DataFrame):
        try:
            logging.info("Starting prediction")

            transformed_feature = self.preprocessing_object.transform(x)

            prediction = self.trained_model_object.predict(transformed_feature)

            return prediction

        except Exception as e:
            raise CustomeException(e, sys)

    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"


# =========================
# Model Trainer
# =========================
class ModelTrainer:

    def __init__(self,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):

        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        self.utils = MainUtils()

    def initiate_model_training(self) -> ModelTrainerArtifact:

        try:
            logging.info("Starting model training")

            # =========================
            # Load Data
            # =========================
            train_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            # =========================
            # Split X and y  ✅ FIXED
            # =========================
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info(f"x_train shape: {x_train.shape}")
            logging.info(f"y_train shape: {y_train.shape}")

            # =========================
            # Model Selection
            # =========================
            model_factory = ModelFactory(
                model_config_path=self.model_trainer_config.model_config_file_path
            )

            best_model_details = model_factory.get_best_model(
                X=x_train,
                y=y_train,
                base_accuracy=self.model_trainer_config.excepted_accuracy
            )

            # =========================
            # Load Preprocessing Objects
            # =========================
            preprocessor_object = self.utils.load_object(
                file_path=self.data_transformation_artifact.transformed_vectorizer_object_file_path
            )

            encoder_object = self.utils.load_object(
                file_path=self.data_transformation_artifact.transformed_encoder_object_file_path
            )

            # =========================
            # Check accuracy
            # =========================
            if best_model_details.best_score < self.model_trainer_config.excepted_accuracy:
                raise Exception("No best model found with required accuracy")

            # =========================
            # Create final model
            # =========================
            spam_model = SpamDetectionModel(
                preprocessing_object=preprocessor_object,
                encoder_object=encoder_object,
                trained_model_object=best_model_details.best_model
            )

            # =========================
            # Save model
            # =========================
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True
            )

            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=spam_model
            )

            logging.info("Model saved successfully")

            # =========================
            # Metrics (dummy for now)
            # =========================
            metric_artifact = ClassificationMetricArtifact(
                f1_score=0.8,
                precision_score=0.8,
                recall_score=0.9
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_path,
                metric_artifact=metric_artifact
            )

            logging.info("Model training completed")

            return model_trainer_artifact

        except Exception as e:
            raise CustomeException(e, sys)
        

if __name__ == "__main__":
    try:
        # =========================
        # Create Data Transformation Artifact
        # =========================
        data_transformation_artifact = DataTransformationArtifact(
            transformed_vectorizer_object_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_24_2026_17_56_33\data_transformation\transformed_object\vectorizer.pkl",

            transformed_encoder_object_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_24_2026_17_56_33\data_transformation\transformed_object\encoder.pkl",

            transformed_train_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_24_2026_17_56_33\data_transformation\transformed\train.npy",

            transformed_test_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_24_2026_17_56_33\data_transformation\transformed\test.npy"
        )

        # =========================
        # Create Model Trainer
        # =========================
        model_trainer = ModelTrainer(
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_config=ModelTrainerConfig()
        )

        # =========================
        # Start Training
        # =========================
        model_trainer_artifact = model_trainer.initiate_model_training()

        print("✅ Model training completed successfully")
        print(model_trainer_artifact)

    except Exception as e:
        raise CustomeException(e, sys)