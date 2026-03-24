from pymongo import MongoClient
import os
from src.constant.env_variable import MONGO_DB_URL
client = MongoClient(os.getenv("MONGO_DB_URL"))

for db in client.list_database_names():
    print("\nDB:", db)
    for col in client[db].list_collection_names():
        print("  Collection:", col,
              "Count:", client[db][col].count_documents({}))