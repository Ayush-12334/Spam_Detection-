import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import CustomeException
from src.logger import logging
from src.constant.database import DATABASE_NAME


class SpamhamData:
    """
    This class helps export the entire MongoDB record as a Pandas DataFrame
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomeException(e, sys)

    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:

        try:
            logging.info("Started fetching the data")

            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomeException(e, sys)