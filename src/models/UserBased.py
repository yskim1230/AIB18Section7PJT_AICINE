import pandas as pd
from surprise import Dataset, Reader, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from surprise.dump import dump
import os

# Define the data folder path
DIR_PATH = "./datasets/"

class UserBasedRecommender:
    def __init__(self, path=None, model_filename=None):
        """
        Initialize the UserBasedRecommender.

        Args:
            path (str): Path to a pre-trained model file (default is None).
            model_filename (str): Path to the file where the trained model will be saved (default is None).
        """
        self.model = None
        self.data_path = os.path.join(DIR_PATH, 'ratings.dat')  # Specify the ratings data path
        self.model_filename = model_filename

        if path is not None:
            self.load_model(path)
        else:
            self.load_data()
            self.select_model()
    
    def save_model(self, model_filename):
        # Save the trained model
        dump(model_filename, algo=self.model)

    def load_data(self):
        # Load ratings data
        ratings = pd.read_csv(self.data_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)

    def select_model(self):
        # Select the collaborative filtering model and configure it
        self.model = KNNBasic(sim_options={'name': 'cosine', 'user_based': True, 'min_support': 5})

    def train_model(self):
        # Train the selected model
        self.model.fit(self.trainset)

    def evaluate_model(self):
        # Make predictions on the test set and calculate RMSE
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        print(f'RMSE: {rmse}')

    def run_pipeline(self):
        # Run the entire pipeline (loading data, selecting model, training, and evaluation)
        self.load_data()
        self.select_model()
        self.train_model()
        self.evaluate_model()

        # Save the trained model if a model filename is provided
        if self.model_filename:
            dump(self.model_filename, algo=self.model)

# Example usage:
# To run the entire pipeline and save the trained model:
pipeline = UserBasedRecommender(model_filename='./models/UserBased.pkl')
pipeline.run_pipeline()
