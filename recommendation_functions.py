import numpy as np
import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

indian_movies = pd.read_csv("indian_movies_dummy.csv")


def recommendation_cosine_similarity(titile_id):
    user_movie = indian_movies[indian_movies["imdb_title_id"] == titile_id]

    language_movie = indian_movies[indian_movies["language"] == user_movie["language"].item()]

    language_movie = language_movie[language_movie["imdb_title_id"] != titile_id]

    user_genre_movie = user_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]
    genre_movie = language_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]

    genre_sim = genre_cosine_similarity(user_genre_movie, genre_movie)
    genre_sim = dict(sorted(genre_sim.items(), key=lambda x: x[1], reverse=True))
    genre_sim_title_id = list(genre_sim.keys())

    language_movie = language_movie[indian_movies["imdb_title_id"].astype(str).str.contains(
        "|".join(genre_sim_title_id[:len(genre_sim_title_id) // 2]))]

    description_dict = description_similarity(user_movie["imdb_title_id"].item(),
                                              language_movie["imdb_title_id"].to_list())
    description_dict = dict(sorted(description_dict.items(), key=lambda x: x[1], reverse=True))
    description_dict_title_id = list(description_dict.keys())

    language_movie = language_movie[indian_movies["imdb_title_id"].astype(str).str.contains(
        "|".join(description_dict_title_id[:len(description_dict_title_id) // 2]))]

    # language_movie = language_movie.sort_values(by="weighted_average_vote", ascending=False)

    return language_movie.head(10)


def genre_cosine_similarity(user_movie, indian_movies):
    genre_sim = {}
    user_movie_array = np.array(user_movie.drop("imdb_title_id", axis=1))
    for i, row in indian_movies.iterrows():
        row_array = np.array(row.drop("imdb_title_id"))
        row_array = row_array.reshape(1, -1)
        sim = cosine_similarity(row_array, user_movie_array)
        genre_sim[row["imdb_title_id"]] = sim[0][0]
    return genre_sim


def description_similarity(user_movie_title_id, indian_movies_title_id):
    description_dict = {}
    with open("description_movie.json", "r") as des_file:
        description_data = json.load(des_file)
    description_data
    user_movie_description_data = np.array(description_data[user_movie_title_id]).reshape(1, -1)
    for i in indian_movies_title_id:
        row_description_data = np.array(description_data[i]).reshape(1, -1)
        row_description_data
        sim = cosine_similarity(row_description_data, user_movie_description_data)
        description_dict[i] = sim[0][0]
    return description_dict


def recommendation_distance_metrics(titile_id, metrics):
    user_movie = indian_movies[indian_movies["imdb_title_id"] == titile_id]

    language_movie = indian_movies[indian_movies["language"] == user_movie["language"].item()]

    language_movie = language_movie[language_movie["imdb_title_id"] != titile_id]

    user_genre_movie = user_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]
    genre_movie = language_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]

    genre_sim = genre_distance_metrics(user_genre_movie, genre_movie, metrics)
    genre_sim = dict(sorted(genre_sim.items(), key=lambda x: x[1], reverse=False))
    genre_sim_title_id = list(genre_sim.keys())

    language_movie = language_movie[indian_movies["imdb_title_id"].astype(str).str.contains(
        "|".join(genre_sim_title_id[:len(genre_sim_title_id) // 2]))]

    description_dict = description_distance_metrics(user_movie["imdb_title_id"].item(),
                                                    language_movie["imdb_title_id"].to_list(), metrics)
    description_dict = dict(sorted(description_dict.items(), key=lambda x: x[1], reverse=False))
    description_dict_title_id = list(description_dict.keys())

    language_movie = language_movie[indian_movies["imdb_title_id"].astype(str).str.contains(
        "|".join(description_dict_title_id[:len(description_dict_title_id) // 2]))]

    language_movie = language_movie.sort_values(by="weighted_average_vote", ascending=False)

    return language_movie.head(10)


def genre_distance_metrics(user_movie, indian_movies, metrics):
    genre_sim = {}
    user_movie_array = np.array(user_movie.drop("imdb_title_id", axis=1))
    for i, row in indian_movies.iterrows():
        row_array = np.array(row.drop("imdb_title_id"))
        row_array = row_array.reshape(1, -1)
        sim = metrics(row_array, user_movie_array)
        genre_sim[row["imdb_title_id"]] = sim[0][0]
    return genre_sim


def description_distance_metrics(user_movie_title_id, indian_movies_title_id, metrics):
    description_dict = {}
    with open("description_movie.json", "r") as des_file:
        description_data = json.load(des_file)
    description_data
    user_movie_description_data = np.array(description_data[user_movie_title_id]).reshape(1, -1)
    for i in indian_movies_title_id:
        row_description_data = np.array(description_data[i]).reshape(1, -1)
        row_description_data
        sim = metrics(row_description_data, user_movie_description_data)
        description_dict[i] = sim[0][0]
    return description_dict


def recommendation_genre_description_cosine(title_id):
    user_movie = indian_movies[indian_movies["imdb_title_id"] == title_id]

    language_movie = indian_movies[indian_movies["language"] == user_movie["language"].item()]

    language_movie = language_movie[language_movie["imdb_title_id"] != title_id]

    user_genre_movie = user_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]
    genre_movie = language_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]

    genre_sim = description_genre_similarity(user_genre_movie, genre_movie)
    genre_sim = dict(sorted(genre_sim.items(), key=lambda x: x[1], reverse=True))
    genre_sim_title_id = [i for i in genre_sim if genre_sim[i] >= 0.9]

    language_movie = language_movie[
        indian_movies["imdb_title_id"].astype(str).str.contains("|".join(genre_sim_title_id))]

    language_movie = language_movie.sort_values(by="weighted_average_vote", ascending=False)

    return language_movie.head(10)


# %%
def description_genre_similarity(user_movie, indian_movies):
    description_dict = {}
    with open("description_movie.json", "r") as des_file:
        description_data = json.load(des_file)
    user_movie_dict = dict(user_movie.drop("imdb_title_id", axis=1))
    # user_movie_dict["description_sparse_matrix"] = description_data[user_movie["imdb_title_id"].item()]
    user_movie_description_data = [user_movie_dict[key].item() for key in user_movie_dict]
    user_movie_description_data = np.array(user_movie_description_data)
    user_movie_description_data = np.append(user_movie_description_data,
                                            description_data[user_movie["imdb_title_id"].item()]).reshape(1, -1)
    for i, row in indian_movies.iterrows():
        row_array = dict(row.drop("imdb_title_id"))
        row_data = [row_array[key] for key in row_array]
        row_data = np.array(row_data)
        row_data = np.append(row_data, description_data[row["imdb_title_id"]]).reshape(1, -1)
        sim = cosine_similarity(row_data, user_movie_description_data)
        description_dict[i] = sim[0][0]
    return description_dict


from scipy.stats import pearsonr


def recommendation_pearson(titile_id):
    user_movie = indian_movies[indian_movies["imdb_title_id"] == titile_id]

    language_movie = indian_movies[indian_movies["language"] == user_movie["language"].item()]

    language_movie = language_movie[language_movie["imdb_title_id"] != titile_id]

    user_genre_movie = user_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]
    genre_movie = language_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]

    genre_sim = genre_pearson(user_genre_movie, genre_movie)
    genre_sim = dict(sorted(genre_sim.items(), key=lambda x: x[1], reverse=True))
    genre_sim_title_id = list(genre_sim.keys())

    language_movie = language_movie[indian_movies["imdb_title_id"].astype(str).str.contains(
        "|".join(genre_sim_title_id[:len(genre_sim_title_id) // 2]))]

    description_dict = description_pearson(user_movie["imdb_title_id"].item(),
                                           language_movie["imdb_title_id"].to_list())
    description_dict = dict(sorted(description_dict.items(), key=lambda x: x[1], reverse=True))
    description_dict_title_id = list(description_dict.keys())

    language_movie = language_movie[indian_movies["imdb_title_id"].astype(str).str.contains(
        "|".join(description_dict_title_id[:len(description_dict_title_id) // 2]))]

    language_movie = language_movie.sort_values(by="weighted_average_vote", ascending=False)

    return language_movie.head(10)


def genre_pearson(user_movie, indian_movies):
    genre_sim = {}
    user_movie_array = np.array(user_movie.drop("imdb_title_id", axis=1)).flatten()
    for i, row in indian_movies.iterrows():
        row_array = np.array(row.drop("imdb_title_id")).flatten()
        corr, p = pearsonr(row_array, user_movie_array)
        if corr < 0:
            genre_sim[row["imdb_title_id"]] = -1 * corr
        else:
            genre_sim[row["imdb_title_id"]] = corr
    return genre_sim


def description_pearson(user_movie_title_id, indian_movies_title_id):
    description_dict = {}
    with open("description_movie.json", "r") as des_file:
        description_data = json.load(des_file)
    description_data
    user_movie_description_data = np.array(description_data[user_movie_title_id]).flatten()
    for i in indian_movies_title_id:
        row_description_data = np.array(description_data[i]).flatten()
        row_description_data
        corr, p = pearsonr(row_description_data, user_movie_description_data)
        if corr < 0:
            description_dict[i] = -1 * corr
        else:
            description_dict[i] = corr
    return description_dict


def recommendation_genre_description_pearson(title_id):
    user_movie = indian_movies[indian_movies["imdb_title_id"] == title_id]

    language_movie = indian_movies[indian_movies["language"] == user_movie["language"].item()]

    language_movie = language_movie[language_movie["imdb_title_id"] != title_id]

    user_genre_movie = user_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]
    genre_movie = language_movie[
        ['imdb_title_id', 'Action', 'Crime', 'Drama', 'Romance', 'Comedy', 'Thriller', 'Musical', 'Adventure',
         'Fantasy', 'History', 'Biography', 'Family', 'Mystery', 'Horror', 'Music', 'War', 'Sport', 'Sci-Fi',
         'Animation']]

    genre_sim = description_genre_similarity_pearson(user_genre_movie, genre_movie)
    genre_sim = dict(sorted(genre_sim.items(), key=lambda x: x[1], reverse=True))
    genre_sim_title_id = [i for i in genre_sim if genre_sim[i] >= 0.9]

    language_movie = language_movie[
        indian_movies["imdb_title_id"].astype(str).str.contains("|".join(genre_sim_title_id))]

    language_movie = language_movie.sort_values(by="weighted_average_vote", ascending=False)

    return language_movie.head(10)


def description_genre_similarity_pearson(user_movie, indian_movies):
    description_dict = {}
    with open("description_movie.json", "r") as des_file:
        description_data = json.load(des_file)
    user_movie_dict = dict(user_movie.drop("imdb_title_id", axis=1))
    # user_movie_dict["description_sparse_matrix"] = description_data[user_movie["imdb_title_id"].item()]
    user_movie_description_data = [user_movie_dict[key].item() for key in user_movie_dict]
    user_movie_description_data = np.array(user_movie_description_data)
    user_movie_description_data = np.append(user_movie_description_data,
                                            description_data[user_movie["imdb_title_id"].item()]).flatten()
    for i, row in indian_movies.iterrows():
        row_array = dict(row.drop("imdb_title_id"))
        row_data = [row_array[key] for key in row_array]
        row_data = np.array(row_data)
        row_data = np.append(row_data, description_data[row["imdb_title_id"]]).flatten()
        corr, p = pearsonr(row_data, user_movie_description_data)
        if corr < 0:
            description_dict[i] = -1 * corr
        else:
            description_dict[i] = corr
    return description_dict


def recommendation_with_list_of_movies(movie_title_list):
    main_movies = pd.DataFrame
    for i in movie_title_list:
        movies = recommendation_cosine_similarity(i)
        movies = pd.concat([movies, recommendation_distance_metrics(i, manhattan_distances)])
        movies = pd.concat([movies, recommendation_distance_metrics(i, euclidean_distances)])
        movies = pd.concat([movies, recommendation_pearson(i)])
        movies = pd.concat([movies, recommendation_genre_description_cosine(i)])
        movies = pd.concat([movies, recommendation_genre_description_pearson(i)])
        movies_names = movies.groupby("imdb_title_id").size().reset_index(name="count")
        movies_names = movies_names.sort_values(by="count", ascending=False, ignore_index=True)
        m = movies_names["imdb_title_id"][:2].to_list()
        if main_movies.empty:
            main_movies = movies[movies["imdb_title_id"].astype(str).str.contains("|".join(m))].drop_duplicates()
        else:
            main_movies = pd.concat(
                [main_movies, movies[movies["imdb_title_id"].astype(str).str.contains("|".join(m))].drop_duplicates()])
    return main_movies.drop_duplicates()


def recommendation_of_movie(titleId):
    movies = recommendation_cosine_similarity(titleId)
    movies = pd.concat([movies, recommendation_distance_metrics(titleId, manhattan_distances)])
    movies = pd.concat([movies, recommendation_distance_metrics(titleId, euclidean_distances)])
    movies = pd.concat([movies, recommendation_pearson(titleId)])
    movies = pd.concat([movies, recommendation_genre_description_cosine(titleId)])
    movies = pd.concat([movies, recommendation_genre_description_pearson(titleId)])
    movies_names = movies.groupby("imdb_title_id").size().reset_index(name="count")
    m = movies_names[movies_names["count"] > 2]["imdb_title_id"].to_list()
    return movies[movies["imdb_title_id"].astype(str).str.contains("|".join(m))].drop_duplicates()
