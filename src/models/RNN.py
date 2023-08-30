import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.layers import SimpleRNN

# RMSE 계산 함수
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

# RNN 기반 추천 시스템 클래스를 정의
class RNNRecommender:
    def __init__(self, num_movies, max_sequence_length, model_path=None):
        # 초기화 메서드에서 여러 변수들을 설정합니다.
        if model_path:
            # Load the model to extract necessary attributes
            temp_model = tf.keras.models.load_model(model_path)
            self.num_movies = temp_model.layers[4].input_dim
            self.max_sequence_length = temp_model.layers[4].input_length
            self.model = temp_model
        else:
            self.num_movies = num_movies
            self.max_sequence_length = max_sequence_length
            self.model = None

    # 추천 시스템의 예측 메서드를 정의합니다.
    def predict(self, user_id, sequence, top=10):
        if self.model is None:
            # 만약 모델이 로드되거나 학습되지 않았다면, 예외를 발생시킵니다.
            raise Exception("Model not loaded or trained.")
        # 사용자의 평점 데이터를 로드합니다.
        ratings = pd.read_csv('./datasets/ratings.dat', sep='::', header=None, engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
        # 지정된 사용자의 평점 데이터만 필터링합니다.
        user_ratings = ratings[ratings['userId'] == user_id]
        # 지정된 사용자가 평가하지 않은 영화들의 목록을 가져옵니다.
        other_movies = ratings[~ratings['movieId'].isin(user_ratings['movieId'])]['movieId'].unique()
        # 지정된 사용자에 대한 예측 점수를 계산합니다.
        predictions = [(movie_id, self.model.predict(user_id, movie_id).est) for movie_id in other_movies]
        # 예측된 점수를 기반으로 상위 영화들을 정렬합니다.
        top_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:top]
        # 상위 영화들의 ID만 반환합니다.
        return [movie[0] for movie in top_movies]

    # 사용자가 최근에 시청한 영화를 반환하는 메서드를 정의합니다.
    def get_recently_watched_movies(self, user_id, ratings, top=5):
        # 지정된 사용자의 평점 데이터만 필터링합니다.
        user_ratings = ratings[ratings['userId'] == user_id]
        # 지정된 사용자의 평점 데이터만 필터링합니다.
        sorted_ratings = user_ratings.sort_values(by='timestamp', ascending=False)
        # 최근에 시청한 영화의 ID만 반환합니다.
        return sorted_ratings['movieId'].head(top).tolist()
    
    # 지정된 사용자에 대한 추천 영화를 반환하는 메서드를 정의합니다.
    def predict_for_user(self, user_id, ratings, movies_df, top=10):
        if self.model is None:
            # 만약 모델이 로드되거나 학습되지 않았다면, 예외를 발생시킵니다.
            raise Exception("Model not loaded or trained.")

        # 지정된 사용자의 평점 데이터만 가져옵니다.
        user_ratings = ratings[ratings['userId'] == user_id]
        # 지정된 사용자가 평가하지 않은 영화들의 목록을 가져옵니다.
        other_movies = ratings[~ratings['movieId'].isin(user_ratings['movieId'])]['movieId'].unique()

        # 각 영화에 대한 예측을 위한 입력 데이터를 생성합니다.
        user_input = np.array([user_id] * len(other_movies))
        movie_input = other_movies
        # 모델을 사용하여 예측 점수를 계산합니다.
        predictions = self.model.predict([user_input, movie_input])
        movie_predictions = list(zip(other_movies, predictions))
        # 예측된 점수를 기반으로 상위 영화들을 정렬합니다.
        top_movies = sorted(movie_predictions, key=lambda x: x[1], reverse=True)[:top]
        # 상위 영화들의 ID만 반환합니다.
        return [movie[0] for movie in top_movies]
    
    # 사용자별로 시청한 영화의 시퀀스를 생성하는 메서드를 정의합니다.    
    def create_sequences(self, ratings):
        # 사용자 ID와 시간 순으로 평점 데이터를 정렬합니다.
        sorted_ratings = ratings.sort_values(by=['userId', 'timestamp'])
        # 각 사용자별로 시청한 영화의 시퀀스를 생성합니다.
        movie_sequences = sorted_ratings.groupby('userId')['movieId'].apply(list).tolist()
        # 가장 긴 시퀀스의 길이를 계산합니다.
        max_sequence_length = max([len(sequence) for sequence in movie_sequences])
        # 모든 시퀀스를 동일한 길이로 패딩합니다.
        movie_sequences = tf.keras.preprocessing.sequence.pad_sequences(movie_sequences, maxlen=max_sequence_length)
        # 생성된 시퀀스와 가장 긴 시퀀스의 길이를 반환합니다.
        return movie_sequences, max_sequence_length


    # 모델을 학습하는 메서드를 정의합니다.
    def train(self, ratings, epochs=5, batch_size=64):
        # 평점 데이터에서 고유한 영화 ID를 추출합니다.
        unique_movie_ids = ratings['movieId'].unique()
        # 영화 ID를 인덱스로 변환하는 딕셔너리를 생성합니다.
        movie2idx = {movie: i for i, movie in enumerate(unique_movie_ids)}
        # 인덱스를 영화 ID로 변환하는 딕셔너리를 생성합니다.
        idx2movie = {i: movie for movie, i in movie2idx.items()}
        # 평점 데이터에 영화 인덱스를 추가합니다.
        ratings['movie_idx'] = ratings['movieId'].map(movie2idx)

        # 각 사용자별로 시청한 영화의 인덱스 시퀀스를 생성합니다.
        user_sequences = ratings.groupby('userId')['movie_idx'].apply(list).reset_index()
        # 시퀀스를 패딩하여 모든 시퀀스가 동일한 길이를 갖도록 합니다.
        user_sequences['movie_idx'] = user_sequences['movie_idx'].apply(lambda x: pad_sequences([x], maxlen=self.max_sequence_length)[0])
        # 패딩된 시퀀스를 추출합니다.
        movie_sequences = pad_sequences(user_sequences['movie_idx'].values, maxlen=self.max_sequence_length)
        # 사용자 ID를 추출합니다.
        user_ids = user_sequences['userId'].values

        # 입력 데이터와 레이블을 정의합니다.
        X = [user_ids, movie_sequences]
        y = ratings.groupby('userId')['rating'].apply(list).apply(lambda x: x[-1]).values

        # RNN 모델의 구조를 정의합니다.
        user_input = Input(shape=(1,))
        movie_input = Input(shape=(self.max_sequence_length,))

        # 영화와 사용자의 임베딩 레이어를 정의합니다.
        user_embed = Embedding(len(user_ids) + 1, 50)(user_input)
        user_embed = tf.keras.layers.Flatten()(user_embed)
        movie_embed = Embedding(self.num_movies, 50)(movie_input)
        
        #rnn_out = LSTM(50)(movie_embed)
        rnn_out = SimpleRNN(50)(movie_embed)
        # 영화와 사용자의 임베딩 레이어를 정의합니다.
        concat = tf.keras.layers.Concatenate()([user_embed, rnn_out])
        # 완전 연결 레이어를 추가합니다.
        x = Dense(128, activation='relu')(concat)
        x = Dropout(0.5)(x)
        # 최종 출력 레이어를 정의합니다.
        outputs = Dense(1)(x)

        # 모델을 구성하고 컴파일합니다.
        self.model = Model(inputs=[user_input, movie_input], outputs=outputs)
        #self.model.compile(optimizer=Adam(), loss='mse')
        self.model.compile(optimizer=Adam(), loss='mse', metrics=[root_mean_squared_error])

        # 데이터를 사용하여 모델을 학습합니다.
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        # 학습된 모델을 저장합니다.
        self.model.save('./models/rnn_recommender.h5')