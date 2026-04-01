import os
import pickle

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.vectorizer_path = os.path.join("artifacts", "vectorizer.pkl")

        if not os.path.exists(self.model_path):
            raise Exception("Model not found. Train first!")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def run_pipeline(self, input_data):
        transformed = self.vectorizer.transform(input_data)
        return self.model.predict(transformed)