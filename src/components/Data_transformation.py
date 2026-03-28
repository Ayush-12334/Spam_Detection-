import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
from imblearn.combine import SMOTETomek
import re

import nltk
from typing import List, Union
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder

from src.exception import CustomeException
from src.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifacts
from src.components.Data_validation import DataValidation
from src.components.Data_ingestion import DataIngestion
from src.constant.training_pipeline import *
from src.entity.config_entity import SimpleImputerConfig
from src.utils.main_utils import MainUtils
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings('ignore')


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifacts,
        data_transformation_config: DataTransformationConfig
    ):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_validation_artifact = data_validation_artifact
        self.data_transformation_config = data_transformation_config
        self.simpleimputer = SimpleImputerConfig()
        self.utils = MainUtils()

    # ✅ FIX 3: renamed from read_file → read_data
    # so initiate_data_transformation's call to read_data works
    @staticmethod
    def read_data(file: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(file, encoding="latin1")
            # keep only columns that don't start with "Unnamed"
            df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
            return df
        except Exception as e:
            raise CustomeException(e, sys) from e

    # ✅ FIX 4: added self so it can be called as self.check_null(df)
    def check_null(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.isnull().values.any():
            df = df.dropna()
        return df  # always return df, even when no nulls found

    def get_limitized_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Entering lemmatization")
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()

            def clean_text(text):
                text = text.lower()
                text = re.sub('[^a-zA-Z]', ' ', text)
                words = word_tokenize(text)
                words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
                return " ".join(words)

            data[FEATURE_COLUMN] = data[FEATURE_COLUMN].apply(clean_text)
            logging.info("Completed lemmatization")
            return data
        except Exception as e:
            raise CustomeException(e, sys) from e

    def get_vectorized_data(
        self,
        traindf: pd.DataFrame,
        testdf: pd.DataFrame,
        vectorizer=None
    ) -> Union[np.ndarray, np.ndarray, object]:
        try:
            if vectorizer is None:
                vectorizer = CountVectorizer()

            logging.info("Entering vectorization")

            # ✅ FIX 4 (continued): self.check_null now works correctly
            checked_train_df = self.check_null(traindf)
            checked_test_df  = self.check_null(testdf)

            # ✅ FIX — logic: lemmatize THEN vectorize (was vectorizing pre-lemmatized text before)
            train_df = self.get_limitized_data(checked_train_df)
            test_df  = self.get_limitized_data(checked_test_df)

            # ✅ FIX — use lemmatized df (train_df/test_df), not checked_df
            vectorized_train = vectorizer.fit_transform(train_df[FEATURE_COLUMN])
            vectorized_test  = vectorizer.transform(test_df[FEATURE_COLUMN])

            logging.info("Train and test df vectorized")

            x_train = vectorized_train.toarray()
            x_test  = vectorized_test.toarray()

            return x_train, x_test, vectorizer

        except Exception as e:
            raise CustomeException(e, sys) from e

    def get_encoded_value(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        encoder=None
    ) -> Union[np.ndarray, np.ndarray, object]:
        try:
            if encoder is None:
                encoder = OrdinalEncoder()

            logging.info("Entering encoding function")

            y_train = train_df[[TARGET_COLUMN]]  # 2D for OrdinalEncoder
            y_test  = test_df[[TARGET_COLUMN]]

            encoded_y_train = encoder.fit_transform(y_train)
            encoded_y_test  = encoder.transform(y_test)

            logging.info("Target column encoded")
            return encoded_y_train, encoded_y_test, encoder

        except Exception as e:
            raise CustomeException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Initiating data transformation")

            # ✅ FIX 3: now calls the correctly named read_data
            train_df = DataTransformation.read_data(file=self.data_ingestion_artifact.trained_file_path)
            test_df  = DataTransformation.read_data(file=self.data_ingestion_artifact.test_file_path)

            # ✅ FIX 1: removed the extra `self,` — was `self, self.get_vectorized_data(...)`
            x_train, x_test, vectorizer = self.get_vectorized_data(traindf=train_df, testdf=test_df)
            y_train, y_test, encoder    = self.get_encoded_value(train_df=train_df, test_df=test_df)

            # create output directories
            preprocessor_obj_dir = os.path.dirname(
                self.data_transformation_config.transformed_vectorizer_object_file_path
            )
            os.makedirs(preprocessor_obj_dir, exist_ok=True)

            encoded_file_path = self.data_transformation_config.transformed_encoded_onject_file_path
            os.makedirs(os.path.dirname(encoded_file_path), exist_ok=True)

            # save encoder
            self.utils.save_object(file_path=encoded_file_path, obj=encoder)

            # ✅ FIX 2: save vectorizer (was saving encoder here by mistake)
            vectorizer_file_path = self.data_transformation_config.transformed_vectorizer_object_file_path
            self.utils.save_object(file_path=vectorizer_file_path, obj=vectorizer)

            # combine features + labels
            train_arr = np.c_[x_train, np.array(y_train)]
            test_arr  = np.c_[x_test,  np.array(y_test)]

            logging.info("Saving transformed train and test arrays")
            self.utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            self.utils.save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            data_transformation_artifact = DataTransformationArtifact(
                transformed_vectorizer_object_file_path=self.data_transformation_config.transformed_vectorizer_object_file_path,
                transformed_encoder_object_file_path=self.data_transformation_config.transformed_encoded_onject_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise CustomeException(e, sys) from e


if __name__ == "__main__":
    try:
        data_ingestion_artifact = DataIngestionArtifact(
            trained_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_19_2026_00_46_41\data_ingestion\ingested\train.csv",
            test_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_19_2026_00_46_41\data_ingestion\ingested\test.csv"
        )

        # ✅ FIX 5: added missing comma between the two path arguments
        data_validation_artifact = DataValidationArtifacts(
            valid_train_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_20_2026_19_44_08\data_ingestion\validated\train.csv",
            valid_test_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_20_2026_19_44_08\data_ingestion\validated\test.csv",
            drift_report_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_20_2026_19_44_08\data_ingestion\drift_report",
            invalid_test_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_20_2026_19_44_08\data_ingestion\invalid",
            invalid_train_file_path=r"C:\Users\Ayush\Machine Learning projects\spam detection\src\artifacts\03_20_2026_19_44_08\data_ingestion\invalid",
            validation_status=True
        )

        data_transformation = DataTransformation(
            data_ingestion_artifact=data_ingestion_artifact,
            data_validation_artifact=data_validation_artifact,
            data_transformation_config=DataTransformationConfig()  # was missing this
        )

        data_transformation.initiate_data_transformation()

    except Exception as e:
        raise CustomeException(e, sys)  