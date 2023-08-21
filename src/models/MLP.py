import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

class MLPRecommender:
    """
    Multi-Layer Perceptron (MLP) based collaborative filtering recommendation model.
    """
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

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
            
        user_ratings = self.data[self.data['userId'] == user_id]
        other_movies = self.data[~self.data['movieId'].isin(user_ratings['movieId'])]['movieId'].unique()
        predictions = [(movie_id, self.model.predict([np.array([user_id]), np.array([movie_id])])[0][0]) for movie_id in other_movies]
        top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top]
        return [movie[0] for movie in top_movies]

    def load_data(self, ratings_path):
        self.data = pd.read_csv(ratings_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])

def train_mlp(ratings_path):
    """
    Train the MLP model and save it to the specified path.

    Args:
        ratings_path (str): Path to the ratings.dat file.
        model_path (str): Path to save the trained model.
    """
    ratings = pd.read_csv(ratings_path, sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user_map = {user: i for i, user in enumerate(user_ids)}
    movie_map = {movie: i for i, movie in enumerate(movie_ids)}
    
    ratings['userId'] = ratings['userId'].map(user_map)
    ratings['movieId'] = ratings['movieId'].map(movie_map)
    
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    
    X = ratings[['userId', 'movieId']].values
    y = ratings['rating'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))

    user_embed = Embedding(num_users, 32)(user_input)
    movie_embed = Embedding(num_movies, 32)(movie_input)

    user_flatten = Flatten()(user_embed)
    movie_flatten = Flatten()(movie_embed)

    concat = Concatenate()([user_flatten, movie_flatten])
    
    dense1 = Dense(256, activation='relu')(concat)
    dense2 = Dense(128, activation='relu')(concat)
    dense3 = Dense(64, activation='relu')(dense1)
    output = Dense(1)(dense3)
    
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    model.fit([X_train[:, 0], X_train[:, 1]], y_train, batch_size=64, epochs=5, validation_data=([X_val[:, 0], X_val[:, 1]], y_val))

    print("---Saving model---")    
    model.save('./models/mlp.h5')
    print('Save Complete.')
