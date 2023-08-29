from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection.search import GridSearchCV

import pandas as pd
import os

import random

class SVDRecommender():
    """
    Singular Value Decomposition (SVD) based collaborative filtering recommendation model.
    """
    def __init__(self, path=None):
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
        

def train(ratings_path, e = [5, 10, 15, 20], lr = [0.001, 0.002, 0.005, 0.01]):
    """
    Train the SVD model and save it to the specified path.
    Args:
        ratings_path (str): Path to the ratings.dat file.
        model_path (str): Path to save the trained model.
    """
    ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    reader = Reader(rating_scale=(1, 5))
    dataset = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    
    # Split the dataset into trainset and testset
    trainset, testset = train_test_split(dataset, test_size=0.2)  # 80% for training, 20% for testing
    
    # Convert trainset back to DataFrame
    trainset_df = pd.DataFrame(trainset.all_ratings(), columns=["userId", "movieId", "rating"])
    trainset_df['userId'] = trainset_df['userId'].apply(lambda x: trainset.to_raw_uid(x))
    trainset_df['movieId'] = trainset_df['movieId'].apply(lambda x: trainset.to_raw_iid(x))
    
    # Convert the DataFrame back to Dataset
    train_dataset = Dataset.load_from_df(trainset_df, reader)
    
    # Select your best algo with grid search.
    print('Grid Search...')
    param_grid = {'n_epochs': e, 'lr_all': lr}
    grid_search = GridSearchCV(SVD, param_grid, measures=['RMSE'])
    grid_search.fit(train_dataset)  # Convert Trainset back to Dataset before passing
    
    print(grid_search.best_params)
    print(grid_search.best_score)

    # Getting the best model from grid search
    best_svd = grid_search.best_estimator['rmse']

    # Training the model on the full dataset
    trainset = dataset.build_full_trainset()
    best_svd.fit(trainset)

    # Making predictions and evaluating on the testset using the best model
    testset = trainset.build_anti_testset()
    predictions = best_svd.test(testset)
    rmse_val = accuracy.rmse(predictions) 
    print(f"RMSE: {rmse_val}")

    #모델 저장
    print("---Saving model---")

    if best_svd is not None:
        from surprise.dump import dump
        dump('./models/svd.h5', algo=best_svd)
    print('Save Complete.')



