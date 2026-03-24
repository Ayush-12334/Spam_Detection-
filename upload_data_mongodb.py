import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from src.constant.env_variable import MONGO_DB_URL
from src.constant.database import DATABASE_NAME, COLLECTION_NAME

load_dotenv()

# 1. Load CSV
csv_file_path = "datasets/spam_ham_dataset.csv"
df = pd.read_csv(csv_file_path, encoding="latin1")

# 2. Connect MongoDB
mongo_url = os.getenv("MONGO_DB")
client = MongoClient(mongo_url)

# 3. Access DB & Collection
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# 4. Clear old data (IMPORTANT)
collection.delete_many({})

# 5. Upload dataset
data = df.to_dict(orient="records")
collection.insert_many(data)

print("Database:", DATABASE_NAME)
print("Collection:", COLLECTION_NAME)
print("Inserted Documents:", collection.count_documents({}))