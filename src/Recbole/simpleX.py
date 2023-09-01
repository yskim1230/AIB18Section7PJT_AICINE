from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.model.general_recommender.simplex import SimpleX
from recbole.trainer import Trainer
import torch

# 모델 설정
config_dict = {
    'data_path': './datasets/',
    'dataset': 'movies',
    'model': 'SimpleX',
    'task': 'point',
    'train': True,
    'gpu_ids': '0',
    'model_num': 1,
    'load_saved_model': False,
    'load_checkpoint_model': False,
    'checkpoint_dir': './saved_model',
    'checkpoint_freq': 1,
    'evaluate_field': ['HitRatio', 'NDCG'],
    'rating_flag': False,
    'weight_decay': 1e-05,
    'learning_rate': 0.001,
    'loss': 'BPR',
    'embedding_size': 64,
    'margin': 0.9,
    'negative_weight': 10,
    'gamma': 0.5,
    'aggregator': 'mean',
    'history_len': 50,
    'reg_weight': 1e-05,
    'early_stopping': True,  # Early Stopping을 사용하려면 True로 설정
    'epochs': 50,  # 학습할 총 에포크 수 설정
    'early_stopping_patience': 5,
}

# 모델 설정 적용
config = Config(config_dict=config_dict)

# 데이터 로더 설정 및 데이터 로드
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

# 모델 초기화
model = SimpleX(config, dataset)

# 트레이너 초기화
trainer = Trainer(config, model)
# 모델 학습
trainer.fit(train_data, valid_data, saved=True, show_progress=True)

# 모델 평가
results = trainer.evaluate(test_data, load_best_model=True)  # Pass test_data for evaluation
print(results)

# 학습된 모델을 저장
model_path = './saved_model/SimpleX.pth'
torch.save(model.state_dict(), model_path)