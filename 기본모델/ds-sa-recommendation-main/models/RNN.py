import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from utils.Dataloader import load_ratings

class RNN:
    """
    RNN based collaborative filtering recommendation model.
    """
    def __init__(self, num_movies, model_path=None):
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self.model = None
        self.num_movies = num_movies

    def predict(self, user_id, sequence, top=10):
        if self.model is None:
            raise Exception("Model not loaded or trained.")
            
        predictions = self.model.predict([np.array([user_id] * self.num_movies), np.array(sequence)])
        sorted_indices = np.argsort(predictions, axis=1)[0][::-1]
        return sorted_indices[:top]

    def load_data(self, ratings_path):
        self.data = load_ratings(ratings_path)

    def train(self, ratings, epochs=5, batch_size=64):
        if ratings is None:
            ratings = Dataloader.load_ratings('datasets')

        user_ids = ratings['userId'].values
        movie_ids = ratings['movieId'].values

        # timestamp 기반으로 가중치를 계산합니다.
        # 이 예에서는 단순히 timestamp를 최대 값으로 나누어 정규화합니다.

        weights = ratings['timestamp'].values / ratings['timestamp'].max()

        X = [user_ids, movie_ids]
        y = ratings['rating'].values

        # Define the RNN model
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(self.num_movies,))
        
        user_embed = Embedding(len(np.unique(user_ids)) + 1, 50)(user_input)
        user_embed = tf.keras.layers.Flatten()(user_embed)  # Flatten layer 추가
        movie_embed = Embedding(self.num_movies + 1, 50)(movie_input)
        
        rnn_out = LSTM(50)(movie_embed)
        concat = tf.keras.layers.Concatenate()([user_embed, rnn_out])
        x = Dense(128, activation='relu')(concat)
        x = Dropout(0.5)(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs=[user_input, movie_input], outputs=outputs)
        self.model.compile(optimizer=Adam(), loss='mse')
        # 학습 시 sample_weight 인자를 통해 가중치를 부여합니다.
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, sample_weight=weights)

        # Save the model
        self.model.save('./models/rnn_recommender.h5')
