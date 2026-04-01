import sys
import os
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from src.exception import CustomeException
from src.logger import logging
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluationArtifact,
    ClassificationMetricArtifact
)
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import MainUtils, load_numpy_array_data


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact
    ):
        self.model_eval_config = model_eval_config
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.utils = MainUtils()

    def evaluate_model(self, model, x, y):
        try:
            print("TYPE OF X:", type(x))
            print("SHAPE OF X:", x.shape)

            y_pred = model.predict(x)

            f1 = f1_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)

            return ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall
            )
        except Exception as e:
            print(" ERROR INSIDE MODEL:", e)
            raise CustomeException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation")

            # load test data
            test_arr = load_numpy_array_data(
                file_path=self.data_transformation_artifact.transformed_test_file_path
            )

            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # load new trained model
            trained_model = self.utils.load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )

            # evaluate new model
            trained_metric = self.evaluate_model(trained_model, x_test, y_test)

            logging.info(f"New Model F1 Score: {trained_metric.f1_score}")

            # check if previous best model exists
            if not os.path.exists(self.model_eval_config.best_model_path):
                logging.info("No previous model found. Accepting new model.")

                os.makedirs(os.path.dirname(self.model_eval_config.best_model_path), exist_ok=True)

                self.utils.save_object(
                    file_path=self.model_eval_config.best_model_path,
                    obj=trained_model
                )

                return ModelEvaluationArtifact(
                    is_model_accepted=True,
                    changed_accuracy=trained_metric.f1_score,
                    best_model_path=self.model_eval_config.best_model_path,
                    trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                    best_model_metric_artifact=trained_metric
                )

            # load previous best model
            best_model = self.utils.load_object(
                file_path=self.model_eval_config.best_model_path
            )

            best_metric = self.evaluate_model(best_model, x_test, y_test)

            logging.info(f"Previous Model F1 Score: {best_metric.f1_score}")

            # compare models
            improvement = trained_metric.f1_score - best_metric.f1_score

            if improvement > self.model_eval_config.changed_threshold_score:
                logging.info("New model is better. Replacing old model.")

                self.utils.save_object(
                    file_path=self.model_eval_config.best_model_path,
                    obj=trained_model
                )

                is_accepted = True
            else:
                logging.info("New model is NOT better.")

                is_accepted = False

            return ModelEvaluationArtifact(
                is_model_accepted=is_accepted,
                changed_accuracy=improvement,
                best_model_path=self.model_eval_config.best_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                best_model_metric_artifact=best_metric
            )

        except Exception as e:
            raise CustomeException(e, sys)