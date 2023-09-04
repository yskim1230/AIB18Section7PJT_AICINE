import argparse
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_scores, full_sort_topk

def main(model_path, user_ids):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_path,
    )

    uid_series = dataset.token2id(dataset.uid_field, user_ids)
    score = full_sort_scores(uid_series, model, test_data, device=config['device'])
    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
    external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

    print('Scores of top 10 items: ', topk_score)
    print('External IDs of top 10 items: ', external_item_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RecBole model and user ID parser')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--user_ids', type=str, nargs='+', required=True, help='List of user IDs')
    
    args = parser.parse_args()

    main(args.model_path, args.user_ids)


# python run.py --model_path ./checkpoints/NARM.pth --user_ids 196 186 
# tensorboard --logdir=./log_tensorboard