import pandas as pd
import numpy as np
from mark_metrics import mark
from collections import defaultdict

def get_user_top_n(user_id, n, model):
    top_n_rec = pd.DataFrame(model.loc[user_id])
    top_n_rec.columns = ['predicted_rating']
    top_n_rec = top_n_rec.sort_values('predicted_rating', ascending=False)
    top_n_rec = top_n_rec.head(n)
    return top_n_rec.index.tolist()


def get_algo_mark(predictions, k_items=10):
    pred_df = pd.DataFrame(predictions)
    pred_df.drop("details", inplace=True, axis=1)
    pred_df.columns = ['userId', 'movieId', 'actual', 'cf_predictions']

    cf_model = pred_df.pivot_table(index='userId', columns='movieId', values='cf_predictions').fillna(0)

    pred_df = pred_df.copy().groupby('userId').agg(
        actual=pd.NamedAgg(column='movieId', aggfunc=(lambda x: list(set(x)))))

    cf_recs = [] = []
    for user_id in pred_df.index:
        cf_predictions = get_user_top_n(user_id, k_items, cf_model)
        cf_recs.append(cf_predictions)

    pred_df['cf_predictions'] = cf_recs

    actual = pred_df.actual.values.tolist()
    cf_pred = pred_df.cf_predictions.values.tolist()

    cf_mark = []
    for K in np.arange(1, k_items + 1):
        cf_mark.extend([mark(actual, cf_pred, k=K)])

    return cf_mark

def get_rating_by_genre(ratings_df, movies_df):
    genre_ratings_dict = defaultdict(list)

    for user_id in ratings_df['userId']:
        rating = ratings_df.loc[user_id]['rating']
        movie_id = ratings_df.loc[user_id]['movieId']
        for genre in movies_df.loc[movie_id]['genres'].split('|'):
            # Append the user rating for a movie to  each associated genre
            genre_ratings_dict[genre].append(rating)

    genre_list = []
    avg_rating_list = []

    for genre, ratings in genre_ratings_dict.items():
        genre_list.append(genre)
        avg_rating_list.append(sum(rating for rating in ratings) / len(ratings))

    return genre_list, avg_rating_list