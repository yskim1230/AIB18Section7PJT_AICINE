import sys
from pathlib import Path
import os
import argparse
from joblib import dump, load

from recbole.config import Config
from recbole.utils.enum_type import ModelType
from recbole.model.general_recommender.lightgcn import LightGCN
from recbole.trainer import Trainer

from recbole.data import create_dataset, data_preparation
from recbole.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils import format_for_recbole, model_pth_handler


def _Relative_Path():
    base_path = Path(os.getcwd())
    current_file_path = Path(FILE.parents[0])
    return current_file_path.relative_to(base_path)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--epochs',type=int, default=0)
    parser.add_argument('-u', '--user', type=int, default=1)
    parser.add_argument('-n', '--num', type=int, default=5)
    args = parser.parse_args()
    return args


def _Config(parents_path=_Relative_Path(), data_path='datasets', dataset='movies', checkpoint_dir= 'saved', epochs=10):
    config_dict = {
        'data_path': os.path.join(parents_path, data_path),                     # config
        'checkpoint_dir': os.path.join(parents_path, checkpoint_dir),           # Config
        'dataset': dataset,                                                     # Config
        'model': 'LightGCN',                                                        # create_dataset, Config
        'MODEL_TYPE': ModelType.GENERAL,                                        # create_dataset
        'learning_rate': 0.005,                                                 # Config
        'train_batch_size': 1024,                                                # training
        'eval_batch_size': 1024,                                                 # training
        'epochs': epochs,                                                       # training
        'n_layers': 4,                                                          # training, The number of layers in LightGCN
        'learner': 'adam',                                                      # training
        'weight_decay': 0.0,                                                    # training, L2 penalty
        'topk': 10,                                                             # training
        'USER_ID_FIELD': 'userId',                                              # dataset config
        'ITEM_ID_FIELD': 'movieId',                                             # dataset config
        'RATING_FIELD': 'rating',                                               # dataset config
        'TIME_FIELD': 'timestamp',                                              # dataset config
        'NEG_PREFIX': 'neg_',                                                   # dataset config
        'load_col': {                                                           # dataset config
            'user': ['userId', 'gender', 'age', 'Occupation', 'zip_code'], 
            'inter': ['userId', 'movieId', 'rating', 'timestamp'],
            'item': ['movieId', 'title', 'genres'],
        }, 
        'saved': True, 
        'verbose': True,                                                        # fit
        'show_progress': True,                                                  # fit, fit 
        'state': 'INFO', 
        'tensorboard': True,
    }
    return config_dict


def recbole_feature_engineering(config_dict=_Config()):
    ratings_df = format_for_recbole.load_ratings_for_recbole(config_dict['data_path'])
    movies_df = format_for_recbole.load_movies_for_recbole(config_dict['data_path'])
    users_df = format_for_recbole.load_users_for_recbole(config_dict['data_path'])

    ratings_df.to_csv(os.path.join(config_dict['data_path'], config_dict['dataset'], 'movies.inter'), index=False, sep='\t')
    movies_df.to_csv(os.path.join(config_dict['data_path'], config_dict['dataset'], 'movies.item'), index=False, sep='\t')
    users_df.to_csv(os.path.join(config_dict['data_path'], config_dict['dataset'], 'movies.user'), index=False, sep='\t')

    return ratings_df, movies_df, users_df


class LightGCN_Model:
    def __init__(self, train=True):
        self.config_dict = _Config(checkpoint_dir= 'saved')
        if not train:
            print("init train False")
            file_paths = model_pth_handler.find_saved_model(model_name=self.config_dict['model'], 
                                                            load_pth_dir_path=self.config_dict["checkpoint_dir"]
                                                            )
            self.file_model_file_name = file_paths[-1]
            print(self.config_dict["checkpoint_dir"])
            print(f"사용할 saved 파일(모델) 경로: {self.file_model_file_name}")
            self.config, self.model, self.dataset, self.train_data, self.valid_data, self.test_data = load_data_and_model(
                model_file=os.path.join(self.config_dict["checkpoint_dir"], self.file_model_file_name), 
                )
        else:
            print("init train True")


    def train(self, epochs=20, del_model=False):
        print("---LightGCN Model Training...---")
        self.config_dict['epochs'] = epochs

        self.config = Config(config_dict=self.config_dict)

        if del_model:
            model_pth_handler.delete_saved_model(model_name=self.config_dict['model'], 
                                                 saved_pth_dir_path=self.config_dict["checkpoint_dir"]
                                                 )

        self.dataset = create_dataset(self.config)
        self.train_data, self.valid_data, self.test_data = data_preparation(self.config, self.dataset)
        self.model = LightGCN(self.config, self.dataset)
        self.trainer = Trainer(self.config, self.model)
        self.trainer.fit(self.train_data, self.valid_data, 
                        saved=self.config['saved'], 
                        verbose=self.config['verbose'],
                        show_progress=self.config['show_progress'],
                        )

        print("---LightGCN Model and Dataset Training Complete---")

    def evaluate(self):
        test_result = self.trainer.evaluate(self.test_data)
        print(f"test_data 평가 점수\n{test_result}")

    def predict(self, user_id=1, n=5):
        uid_series = self.dataset.token2id(self.dataset.uid_field, [str(user_id)])
        _, topk_iid_list = full_sort_topk(uid_series, self.model, self.test_data, k=n, device=self.config['device'])
        external_item_list = self.dataset.id2token(self.dataset.iid_field, topk_iid_list.cpu())

        return list(external_item_list[0])


if __name__ == '__main__':
    args = parse_opt()
    if args.train:
        print("args.train True")
        relative_path = _Relative_Path()                    # 작업 디렉토리
        print(f"작업 경로{relative_path}")

        ract = LightGCN_Model()
        ract.train(epochs=args.epochs, 
                   del_model=True,
                                )
        ract.evaluate()
        print(ract.predict())
    elif args.user:
        print("args.train False, args.user True")
        config_dict = _Config()
        print(config_dict)

        ract = LightGCN_Model(train=False)
        print(ract.predict(user_id=args.user, n=args.num))
    else:
        print("fail")
