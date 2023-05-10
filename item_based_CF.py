import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# Load the MovieLens 1M dataset
ratings_df = pd.read_csv('dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
movies_df = pd.read_csv('dataset/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python')
users_df = pd.read_csv("dataset/users.dat", sep="::", names=["user_id", "gender", "age", "occupation", "zipcode"], engine='python')

print(movies_df.head())

# Encodage de la colonne "movie_id"
movies_df["movie_id"] = pd.Categorical(movies_df["movie_id"]).codes

# Encodage de la colonne "title"
movies_df["title"] = pd.Categorical(movies_df["title"]).codes

#print(movies_df.head())

# Encodage de la colonne "genres"
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies_df["genres"].str.split("|"))
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
movies_df = pd.concat([movies_df, genres_df], axis=1).drop("genres", axis=1)

#print(movies_df.head())
#print(movies_df.columns)

# Encodage de la colonne "user_id"
ratings_df["user_id"] = pd.Categorical(ratings_df["user_id"]).codes

# Encodage de la colonne "movie_id"
ratings_df["movie_id"] = pd.Categorical(ratings_df["movie_id"]).codes

#print(ratings_df.head())

# Encodage de la colonne "gender"
le = LabelEncoder()
users_df["gender"] = le.fit_transform(users_df["gender"])

# Encodage de la colonne "user_id"
users_df["user_id"] = le.fit_transform(users_df["user_id"])

#print(users_df.head())

#fusionner les fichiers
ratings_movies = pd.merge(ratings_df, movies_df, on='movie_id')
data = pd.merge(ratings_movies, users_df, on='user_id')

userRatings = ratings_movies.pivot_table(index=['user_id'],columns=['title'],values='rating')
print(userRatings.head())
#print(data.head())

#suppression des colonnes inutiles
data.drop(['timestamp', 'zipcode'], axis=1, inplace=True)

#print(data.columns)

# Supprimer les lignes en double
data = data.drop_duplicates()

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Identifier les valeurs aberrantes
boxplot = data.boxplot(column=["rating"])
