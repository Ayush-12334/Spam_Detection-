import sys 
from typing import Tuple
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constant.database import DATABASE_NAME,COLLECTION_NAME
from src.exception import CustomeException
from src.logger import logging
from src.data_acess.spamham_data import  SpamhamData
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.utils.main_utils import MainUtils
from src.constant.training_pipeline import *







class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig =DataIngestionConfig()):

        self.data_ingestion_config=data_ingestion_config
        self.utils=MainUtils()

    def split_data_as_train_test(self,dataframe :pd.DataFrame) ->Tuple[pd.DataFrame,pd.DataFrame]:


        '''
        Method Name : split_data_as_train_test
        Description : This method splits the dataframe into Train set and test set based on split ratio

        Output      : Folder is created locally and files are stored
        On Failure  : Write an Exception  log and then raise and Exception 

        Version     : 1.0.0
        Revision    : Moved setup to cloude


        '''
        try:

            logging.info("entered into data train and test on dataframe")

            train_set,test_set=train_test_split(dataframe,train_size=self.data_ingestion_config.train_test_split_ratio)

            logging.info("Done with the split phase ")



            ingested_data_dir=self.data_ingestion_config.ingested_data_dir
            os.makedirs(ingested_data_dir,exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            logging.info("Training data has been  saved")
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info('Test data has been saved')


        except Exception as e:
            raise CustomeException(e,sys) from e
        
        


    def export_data_into_feature_store(self)->pd.DataFrame:
        """
        Method_Name : Export_data_into_feature_store
        Description : This methoad reads data from mongodb and  save into artifacts

        OUTPUT      : Dataset is returned as Dataframe
        on Failure  : write an exception log and then raise an exception 

        version     : 1.0.0
        """


        try:
            logging.info("reading the data from mongo db")
            customer_data =SpamhamData()
            customer_dataframe= customer_data.export_collection_as_dataframe(
                collection_name=COLLECTION_NAME
            )
            logging.info(f"DATA_INGESTION:shape of dataframe{customer_dataframe.shape}")
            feature_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"saving the exported data {feature_file_path}")
            customer_dataframe.to_csv(feature_file_path,index=False,header=True)
            return customer_dataframe   
        
        except Exception as e:
            raise CustomeException(e,sys) from e
        


    def initiate_data_ingestion(self)->DataIngestionArtifact:
        try:

            logging.info("DATA_INGESTION :Entered into data ingestion  process")

            dataframe=self.export_data_into_feature_store()

            logging.info(" DATA_INGESTION: got the data from mongo db")

            self.split_data_as_train_test(dataframe)

            logging.info("DATA_INGESTION: performed trian_test_split_on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )
            data_ingestion_artifact=DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f'Data ingestion artifacts :{data_ingestion_artifact}')
            return data_ingestion_artifact

        except Exception as e:
            raise CustomeException(e,sys) from e
                      
if __name__== '__main__':
    try:

        data_ingestion=DataIngestion()
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        raise CustomeException(e,sys) from e




