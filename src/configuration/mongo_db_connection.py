import os
import sys

import certifi
import pymongo
from dotenv import load_dotenv

from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGO_DB_URL

from src.exception import CustomeException
from src.logger import logging


ca = certifi.where()

load_dotenv()
class MongoDBClient:
    client = None

    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:

            logging.info('trying to connect')
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGO_DB_URL)
                if mongo_db_url is None:
                    raise Exception(f"Environment key: {MONGO_DB_URL} is not set.")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            
           
            logging.info('connected sucessfully')
        except Exception as e:

            raise CustomeException(e, sys)

           

if __name__ == "__main__":
    client = MongoDBClient()
    
    # 1. Check the actual Cluster Address
    print(f"--- Connection Details ---")
    print(f"Server Address: {client.client.address}") 
    print(f"Database Name: {client.database_name}")
    
    # 2. Check Collections
    db = client.client[client.database_name]
    collections = db.list_collection_names()
    print(f"Collections found: {collections}")
    
    # 3. Count and Peek
    if 'spam_ham_collection' in collections:
        col = db['spam_ham_collection']
        count = col.count_documents({})
        print(f"Document Count: {count}")
        if count > 0:
            print("First Document Sample:", col.find_one())
    else:
        print("CRITICAL: 'spam_ham_collection' NOT found in this database!")
        