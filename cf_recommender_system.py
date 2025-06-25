import pandas as pd
from collections import defaultdict
from surprise import Dataset, SVDpp, Reader
from surprise.model_selection import train_test_split
from eval_functions import get_algo_mark, get_rating_by_genre

class RecSys:

    def __init__(self):
        # Defining instance variables
        self.movies_df = None
        self.ratings_df = None
        self.mark_scores = None
        self.names = None
        self.reader = None
        self.data = None
        self.rec_sys_train_set = None
        self.rec_sys_algo = None
        self.rec_sys_test_set = None
        self.rec_sys_predictions = None
        self.top_n_df = None
        self.all_ratings = None
        self.genre_list = None
        self.rating_list = None

    def set_movie_rating_df(self, m_df, r_df):
        self.movies_df = m_df
        self.ratings_df = r_df

    def make_datasets(self):
        # Building Reader and Dataset for training algos
        self.reader = Reader(rating_scale=(1.0, 5.0))
        self.data = Dataset.load_from_df(self.ratings_df[['userId', 'movieId', 'rating']], self.reader)

        # Building datasets for visualizations
        self.all_ratings = self.ratings_df['rating'].tolist()
        movies_indexed_df = self.movies_df.set_index('movieId')
        self.genre_list, self.rating_list = get_rating_by_genre(self.ratings_df, movies_indexed_df)

    def train_eval_algos(self):
        eval_train_set, eval_test_set = train_test_split(self.data, test_size=0.20)

        # Training algos with different learning rates for validation and comparison
        eval_algo = SVDpp( cache_ratings=True)
        eval_algo.fit(eval_train_set)
        eval_predictions = eval_algo.test(eval_test_set)
        eval_mark = get_algo_mark(eval_predictions, k_items=10)

        self.mark_scores = [eval_mark]
        self.names = ['SVD++']

    def train_rec_sys_algo(self):
        # Training Recommender System
        self.rec_sys_train_set = self.data.build_full_trainset()
        self.rec_sys_algo = SVDpp(cache_ratings=True)
        self.rec_sys_algo.fit(self.rec_sys_train_set)
        self.rec_sys_test_set = self.rec_sys_train_set.build_anti_testset()
        self.rec_sys_predictions = self.rec_sys_algo.test(self.rec_sys_test_set)

        # Making top-n recommendation lists for every user
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.rec_sys_predictions:
            top_n[uid].append((iid, est))
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = [iid for iid, _ in user_ratings[:10]]
        self.top_n_df = pd.DataFrame(
            [(k, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9]) for k, v in top_n.items()],
            columns=['userId',
                     'movie_1', 'movie_2', 'movie_3', 'movie_4', 'movie_5', 'movie_6',
                     'movie_7', 'movie_8', 'movie_9', 'movie_10'])

    def get_top_n_df(self):
        return self.top_n_df