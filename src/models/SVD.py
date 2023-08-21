from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

import pandas as pd
import os

class SVDRecommender():
    """
    Singular Value Decomposition (SVD) based collaborative filtering recommendation model.
    """
    def __init__(self, path=None):
        self.model = None
        if path is not None and os.path.exists(path):
            self.load_model(path)

    def predict(self, user_id, top=10):
        """
        Recommend movies based on the given user ID.

        Args:
            user_id (int): Target user ID for recommendation.
            top (int): Number of top recommended movies to return.

        Returns:
            list: List of top recommended movie IDs.
        """
        if self.model is None:
            raise Exception("Model not loaded or trained.")

        ratings = pd.read_csv('./datasets/ratings.dat', sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        user_ratings = ratings[ratings['userId'] == user_id]
        other_movies = ratings[~ratings['movieId'].isin(user_ratings['movieId'])]['movieId'].unique()
        predictions = [(movie_id, self.model.predict(user_id, movie_id).est) for movie_id in other_movies]
        top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top]
        return [movie[0] for movie in top_movies]

    def save_model(self, model_path):
        """
        Save the trained model to the specified path.

        Args:
            model_path (str): Path to save the trained model.
        """
        if self.model is not None:
            from surprise.dump import dump
            dump(model_path, algo=self.model)

    def load_model(self, model_path):
        """
        Load the pre-trained model from the specified path.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        if os.path.exists(model_path):
            from surprise.dump import load
            _, self.model = load(model_path)
        else:
            raise Exception(f"Model file not found: {model_path}")
        

def train(ratings_path):
    """
    Train the SVD model and save it to the specified path.

    Args:
        ratings_path (str): Path to the ratings.dat file.
        model_path (str): Path to save the trained model.
    """
    ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(dataset, test_size=0.2)

    svd = SVD()
    svd.fit(trainset)
    predictions = svd.test(testset)
    print(f"RMSE: {accuracy.rmse(predictions)}")

    #모델 저장
    print("---Saving model---")

    if svd is not None:
        from surprise.dump import dump
        dump('./models/svd.h5', algo=svd)

    print('Save Complete.')



