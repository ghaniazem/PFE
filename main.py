import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
users = pd.read_csv("dataset/users.dat", sep="::", names=["user_id", "gender", "age", "occupation", "zipcode"], engine='python')
movies = pd.read_csv("dataset/movies.dat", sep="::", names=["movie_id", "title", "genres"], engine='python')
ratings = pd.read_csv('dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')


# Afficher les premières lignes du DataFrame
print(users.head())
print(movies.head())
print(ratings.head())
