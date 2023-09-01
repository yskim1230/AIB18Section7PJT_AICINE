import pandas as pd
from surprise import Dataset, Reader, KNNWithZScore, accuracy
from surprise.model_selection import train_test_split
import pickle
import os

# Define the data folder path
DIR_PATH = "./datasets/"

class ItemBasedRecommender:
    def __init__(self, path=None):
        self.model = None
        self.data_path = os.path.join(DIR_PATH, 'ratings.dat')

        if path is not None:
            self.load_model(path)
        else:
            self.load_data()
            self.select_model()

    def load_data(self):
        ratings = pd.read_csv(self.data_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        self.trainset, self.testset = train_test_split(data, test_size=0.2, random_state=42)

    def select_model(self):
        self.model = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': False, 'min_support': 5})

    def train_model(self):
        self.model.fit(self.trainset)

    def evaluate_model(self):
        predictions = self.model.test(self.testset)
        rmse = accuracy.rmse(predictions)
        print(f'RMSE: {rmse}')

    def run_pipeline(self):
        self.load_data()
        self.select_model()
        self.train_model()
        self.evaluate_model()

    def save_model(self, model_filename):
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.model, model_file)

def train_and_save_model(ratings_path, model_filename):
    reader = Reader(rating_scale=(1, 5))
    ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = KNNWithZScore(sim_options={'name': 'cosine', 'user_based': False, 'min_support': 5})
    model.fit(trainset)
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)

# Example usage:
# To run the entire pipeline for item-based filtering:
pipeline = ItemBasedRecommender()
pipeline.run_pipeline()

# To train and save a new item-based model:
train_and_save_model('./datasets/ratings.dat', './models/ItemBased.pkl')
