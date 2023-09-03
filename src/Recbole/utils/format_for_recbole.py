import os
import sys
from pathlib import Path
import pandas as pd


#경로 설정
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH


def load_ratings_for_recbole(path='./datasets/'):
    COL_NAME = ['userId','movieId','rating','timestamp']
    df = pd.read_csv(os.path.join(path,"ratings.dat"),sep='::', header=None, engine='python', names=COL_NAME)
    df = df.rename(columns={'userId': 'userId:token', 
                            'movieId': 'movieId:token', 
                            'rating': 'rating:float', 
                            'timestamp': 'timestamp:float'}
                            )
    return df


def load_movies_for_recbole(path='./datasets/'):
    COL_NAME = ['movieId','title','genres']
    df = pd.read_csv(os.path.join(path,"movies.dat"),sep='::', header=None, engine='python', names=COL_NAME, encoding = 'ISO-8859-1' )
    df = df.rename(columns={'movieId': 'movieId:token', 
                            'title': 'title:token_seq', 
                            'genres': 'genres:token_seq'}
                            )
    return df


def load_users_for_recbole(path='./datasets/'):
    COL_NAME = ['userId','gender','age','Occupation','zip_code']
    df = pd.read_csv(os.path.join(path,"users.dat"),sep='::', header=None, engine='python', names=COL_NAME)
    df = df.rename(columns={'userId': 'userId:token', 
                            'gender': 'gender:token', 
                            'age': 'age:token', 
                            'Occupation': 'Occupation:token', 
                            'zip_code': 'zip_code:token'}
                            )
    return df