import sys
from pathlib import Path
import os

#경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils import Dataloader
from models import MF, KNN, SVD, MLP, LightFM
import argparse

def str2list(value):
    """Convert a string representation of list to a list."""
    return [float(i) for i in value.strip('[]').split(',')]

def str2list_int(value):
    """Convert a string representation of list to a list of integers."""
    return [int(i) for i in value.strip('[]').split(',')]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--vector_size', type=int, default=100)
    parser.add_argument('--pretrained', type=str, default='glove-twitter-100')
    parser.add_argument('--knn', action=argparse.BooleanOptionalAction)
    parser.set_defaults(knn=True)
    parser.add_argument('--mf', action=argparse.BooleanOptionalAction)
    parser.set_defaults(mf=True)
    parser.add_argument('--lightfm', action=argparse.BooleanOptionalAction)
    parser.set_defaults(lightfm=True)
    parser.add_argument('-k', type=int, default=200)
    parser.add_argument('-e', '--epochs',type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('--svd', action=argparse.BooleanOptionalAction)
    parser.add_argument('-n_e', '--n_epochs', type=str2list_int, default="[5, 10, 15, 20]")
    parser.add_argument('-lr', '--learning_rate', type=str2list, default="[0.001, 0.002, 0.005, 0.01]")
    parser.set_defaults(svd=True)
    parser.add_argument('--mlp', action=argparse.BooleanOptionalAction)
    parser.set_defaults(mlp=True)
    args = parser.parse_args()
    return args

args = parse_args()

#데이터 폴더 경로
DIR_PATH = "./datasets/"

#데이터 호출
users_df = Dataloader.load_users(DIR_PATH)
ratings_df = Dataloader.load_ratings(DIR_PATH)
movies_df = Dataloader.load_movies(DIR_PATH)

if args.mf:
    MF.train(users_df=users_df,
             movies_df=movies_df,
             ratings_df=ratings_df,
             K=args.k,
             epochs=args.epochs,
             batch_size=args.batch_size
            )

if args.knn:
    KNN.train(movies_df=movies_df,
              vector_size = args.vector_size,
              pretrained= args.pretrained
             )
    
if args.svd:
    SVD.train(ratings_path='./datasets/ratings.dat',
              e=args.n_epochs,
              lr=args.learning_rate
             )

if args.mlp:
    MLP.train_mlp(ratings_path='./datasets/ratings.dat'
             )

if args.lightfm:
    LightFM.train(epochs=args.epochs, 
                  num_threads=4
                  )