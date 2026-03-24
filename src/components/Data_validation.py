import os
import sys
from typing import Tuple
import pandas as pd

from src.exception import CustomeException
from src.logger import logging
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifacts
from src.entity.config_entity import DataValidationConfig
from src.utils.main_utils import MainUtils
from src.components.Data_ingestion import DataIngestion
#from evidently.model_profile import Profile
#from evidently.model_profile.sections import DataDriftProfileSection


class DataValidation:

    def __init__(self,
                 data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):

        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.utils = MainUtils()

            self._schema_config = self.utils.read_schema_config_file()

        except Exception as e:
            raise CustomeException(e, sys)

    def validate_schema_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate column schema of dataframe
        """
        try:
       

            status = len(dataframe.columns) == len(self._schema_config['columns'])

            logging.info(f"Schema validation status: {status}")
            return status

        except Exception as e:
            raise CustomeException(e, sys)

    def validate_dataset_schema_columns(self,
                                        train_set: pd.DataFrame,
                                        test_set: pd.DataFrame) -> Tuple[bool, bool]:

        try:
            logging.info("Validating dataset schema")

            train_status = self.validate_schema_columns(train_set)
            test_status = self.validate_schema_columns(test_set)

            return train_status, test_status

        except Exception as e:
            raise CustomeException(e, sys)
        


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:

        try:
            df=pd.read_csv(file_path,encoding='latin1')
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            return df
        except Exception as e:
            raise CustomeException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifacts:
        try:
            logging.info("Starting data validation")

            train_df, test_df = (DataValidation.read_data(file_path = self.data_ingestion_artifact.trained_file_path),
                                DataValidation.read_data(file_path = self.data_ingestion_artifact.test_file_path))
            
            train_status, test_status = self.validate_dataset_schema_columns(
                train_set=train_df,
                test_set=test_df
            )
              # 🔥 CREATE DIRECTORIES FIRST
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            validation_status = train_status and test_status
            if validation_status:
                train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)
            else:
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)

            logging.info(f"Train schema status: {train_status}, Test schema status: {test_status}")


            data_validation_artifact = DataValidationArtifacts(
                validation_status=validation_status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomeException(e, sys)
if __name__ == "__main__":
    try:
        # 🔹 Step 1: Manually create ingestion artifact
        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_19_2026_00_46_41\data_ingestion\ingested\train.csv",
            test_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_19_2026_00_46_41\data_ingestion\ingested\test.csv"
        )

        # 🔹 Step 2: Create config
        data_validation_config = DataValidationConfig()

        # 🔹 Step 3: Run validation
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )

        data_validation_artifact = data_validation.initiate_data_validation()

        print(data_validation_artifact)

    except Exception as e:
        raise CustomeException(e, sys)
