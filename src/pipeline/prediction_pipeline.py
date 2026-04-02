import os
import pickle

class PredictionPipeline:
    def __init__(self, model_path, vectorizer_path):

        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        if not os.path.exists(self.model_path):
            raise Exception(f"Model not found at: {self.model_path}")

        if not os.path.exists(self.vectorizer_path):
            raise Exception(f"Vectorizer not found at: {self.vectorizer_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def run_pipeline(self, input_data):

        if isinstance(input_data, str):
            input_data = [input_data]

        transformed = self.vectorizer.transform(input_data)
        prediction = self.model.predict(transformed)

        return prediction