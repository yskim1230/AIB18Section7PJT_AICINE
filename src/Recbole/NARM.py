from recbole.data import create_dataset, data_preparation
from recbole.config import Config
from recbole.model.sequential_recommender import NARM
from recbole.trainer import Trainer
from recbole.utils.enum_type import ModelType
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk

# Define config for NARM with recall@10 and ndcg@10 as evaluation metrics

class NARMRecommend():
    def __init__(self, path):
        self.config, self.model, self.dataset, self.train_data, self.valid_data, self.test_data = load_data_and_model(model_file=path)
        
    def predict(self, userid, top = 10):
        uid_series = self.dataset.token2id(self.dataset.uid_field, [f'{userid}'])
        topk_score, topk_iid_list = full_sort_topk(uid_series, self.model, self.test_data, k=top, device=self.config['device'])
        external_item_list = self.dataset.id2token(self.dataset.iid_field, topk_iid_list.cpu())

        print('list of top 10 movies : ', external_item_list)
        print('scores of top 10 movies : ', topk_score)

def train():
    config_dict = {
        'data_path': './datasets',  # Path to your dataset
        'checkpoint_dir': './checkpoints',
        'MODEL_TYPE': ModelType.SEQUENTIAL,
        'model': 'NARM',
        'learning_rate': 0.001,
        'epochs': 20,
        'eval_step': 1,
        'stopping_step': 3,
        'batch_size': 2048,
        'state': 'INFO',
        'tensorboard': True,
        'load_col': {
            'inter': ['user_id', 'item_id', 'rating', 'timestamp']
        },
        'train_batch_size': 2048,
        'eval_batch_size': 4096,
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'RATING_FIELD': 'rating',
        'TIME_FIELD': 'timestamp',
        'NEG_PREFIX': 'neg_',
        'eval_setting': 'RO_RS,full',
        'topk': 10,
        'loss_type': 'BPR',  # Cross-Entropy loss
        'metrics': ['Recall', 'NDCG', 'MRR'],
        'dataset': 'movies'  
    }

    config = Config(config_dict=config_dict)

    # Prepare dataset
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # Initialize the model
    model = NARM(config, dataset)

    # Train the model
    trainer = Trainer(config, model)
    print("Training started.")
    trainer.fit(train_data, valid_data, saved=True, show_progress=True)
    print("Training end.")

#if __name__ == "__main__":
#    train()
    # After training the model
    #test_result = trainer.evaluate(test_data, show_progress=True)

    # Print the evaluation results
    #for metric, value in test_result.items():
        #print(f"{metric}: {value:.4f}")

