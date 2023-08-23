import joblib
from utils.Dataloader import load_ratings, load_movies, load_users
import sys
from pathlib import Path
import argparse

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k
import numpy as np


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs',type=int, default=20)
    opt = parser.parse_args()
    return opt


def LFM_feature_engineering(users_df=None, movies_df=None, ratings_df=None):
    if users_df is None:
        users_df = load_users('datasets/')
    if movies_df is None:
        movies_df = load_movies('datasets/')
    if ratings_df is None:
        ratings_df = load_ratings('datasets/')
    
    # ... [기존의 피처 엔지니어링 코드]
    # ratings_source
    ratings_source = [(ratings_df['userId'][i], ratings_df['movieId'][i], ratings_df['rating'][i]) for i in range(ratings_df.shape[0])]

    # ratings_df에서 각 movieId에 대한 평점의 평균을 계산합니다.
    avg_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index().rename(columns={'rating': 'rating_avg'})
    # 평균 평점 정보를 recommended_movies 데이터프레임에 병합합니다.
    movies_avg_ratings = movies_df.merge(avg_ratings, on='movieId', how='left')
    movies_avg_ratings = movies_avg_ratings[['movieId', 'title', 'rating_avg', 'genres']]

    # 사용자 행렬
    def create_user_feature_list(row):
        gender = f"gender:{row['gender']}"
        age = f"age:{row['age']}"
        occupation = f"occupation:{row['Occupation']}"
        zip_code = f"zip_code:{row['zip_code']}"
        return [gender, age, occupation, zip_code]

    user_features_source = [(users_df['userId'][i], create_user_feature_list(users_df.iloc[i])) for i in range(users_df.shape[0])]

    # 아이템 행렬
    def create_item_feature_list(row):
        title = f"title:{row['title']}"
        rating_avg = f"rating_avg:{row['rating_avg']}"
        genres = [f"genre:{genre}" for genre in row['genres'].split('|')] # 가정: 장르가 '|'로 구분됩니다.
        return [title, rating_avg] + genres

    item_features_source = [(movies_avg_ratings['movieId'][i], create_item_feature_list(movies_avg_ratings.iloc[i])) for i in range(movies_avg_ratings.shape[0])]

    # Dataset 객체 생성
    dataset = Dataset()

    # 모든 사용자 ID와 모든 아이템 ID 집합 생성
    all_user_ids = set(x[0] for x in ratings_source).union(set(x[0] for x in user_features_source))
    all_item_ids = set(x[1] for x in ratings_source).union(set(x[0] for x in item_features_source))

    # Dataset 객체에 모든 사용자 ID와 아이템 ID 적용
    dataset.fit(all_user_ids, all_item_ids)

    # 사용자와 아이템의 특성 정보를 fit_partial 메서드에 적용
    dataset.fit_partial(
        user_features=(feature for _, features in user_features_source for feature in features), 
        item_features=(feature for _, features in item_features_source for feature in features)
    )

    # 상호 작용 행렬 생성
    (interactions_matrix, weights_matrix) = dataset.build_interactions((x[0], x[1], x[2]) for x in ratings_source)

    # 아이템 특성 행렬 생성
    item_features_matrix = dataset.build_item_features(item_features_source)

    # 사용자 특성 행렬 생성
    user_features_matrix = dataset.build_user_features(user_features_source)

    return interactions_matrix, item_features_matrix, user_features_matrix


class LightFM_Model:
    def __init__(self, model_path=None, dataset_path=None):
        if model_path is not None:
            self.model = joblib.load(model_path)
        else:
            self.model = LightFM(loss='warp')  # 'warp': Weighted Approximate-Rank Pairwise loss

        if dataset_path is not None:
            self.interactions_matrix, self.item_features_matrix, self.user_features_matrix = joblib.load(dataset_path)
        else:
            self.interactions_matrix, self.item_features_matrix, self.user_features_matrix = LFM_feature_engineering()
    
    def predict(self, user_id, n):
        scores = self.model.predict(user_id, np.arange(self.interactions_matrix.shape[1]),
                                    item_features=self.item_features_matrix,
                                    user_features=self.user_features_matrix)
        top_items = np.argsort(-scores)
        return top_items[:n]

    def evaluate(self):
        train_auc = auc_score(self.model, self.interactions_matrix, 
                              item_features=self.item_features_matrix, 
                              user_features=self.user_features_matrix).mean()
        print(f'Training set AUC: {train_auc}')


def train(epochs=20, num_threads=4):
    print("---LightFM Model Training...---")
    lm = LightFM_Model()
    lm.model.fit(lm.interactions_matrix, 
                 item_features=lm.item_features_matrix,
                 user_features=lm.user_features_matrix,
                 epochs=epochs,
                 num_threads=num_threads, 
                 verbose=2
                 )
    
    lm.evaluate()
    print("---LightFM Model and Dataset Training Complete---")

    joblib.dump(lm.model, "./models/lightfm_model.joblib")
    joblib.dump((lm.interactions_matrix, lm.item_features_matrix, lm.user_features_matrix), "./models/lightfm_dataset.joblib")


if __name__ == '__main__':
    opt = parse_opt()
    # Training
    train(epochs=opt.epochs)

    # Load model and Predict
    lm = LightFM_Model(model_path="./models/lightfm_model.joblib", dataset_path="./models/lightfm_dataset.joblib")

    # 예측 및 평가 코드 수정
    print(lm.predict(user_id=1, n=10))  # 예측 상위 10개 아이템을 출력
    lm.evaluate()
