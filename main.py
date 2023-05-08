import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
users_df = pd.read_csv("dataset/users.dat", sep="::", names=["user_id", "gender", "age", "occupation", "zipcode"], engine='python')
movies_df = pd.read_csv("dataset/movies.dat", sep="::", names=["movie_id", "title", "genres"], engine='python')
ratings_df = pd.read_csv('dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')

#afficher les en-tete de data frames
print("Utilisateurs : \n ",users_df.head(), '\n \n')
print("Filmes : \n ",movies_df.head(), '\n \n')
print("Interactions : \n ",ratings_df.head(), '\n \n \n ')


# Calculer le nombre d'utilisateurs, le nombre d'articles et le nombre d'interactions
num_users = ratings_df['user_id'].nunique()
num_items = ratings_df['movie_id'].nunique()
num_interactions = len(ratings_df)

# Calculer la note moyenne
mean_rating = ratings_df['rating'].mean()

# Calculer les interactions par utilisateur (moyenne, médiane)
user_interactions = ratings_df.groupby('user_id')['movie_id'].count()
user_mean_interactions = user_interactions.mean()
user_median_interactions = user_interactions.median()

# Calculer les interactions par item (moyenne, médiane)
item_interactions = ratings_df.groupby('movie_id')['user_id'].count()
item_mean_interactions = item_interactions.mean()
item_median_interactions = item_interactions.median()

# Afficher les résultats
print(f'Nombre d\'utilisateurs : {num_users}')
print(f'Nombre d\'articles : {num_items}')
print(f'Nombre d\'interactions : {num_interactions}')
print(f'Note moyenne : {mean_rating}')
print(f'Interactions par utilisateur (moyenne) : {user_mean_interactions}')
print(f'Interactions par utilisateur (médiane) : {user_median_interactions}')
print(f'Interactions par article (moyenne) : {item_mean_interactions}')
print(f'Interactions par article (médiane) : {item_median_interactions}', '\n \n \n ')


#----------------Graphiques-------------------

# Calcul du nombre d'évaluations par utilisateur

user_interactions = ratings_df.groupby('user_id')['rating'].count()
print(user_interactions)
# création de la figure
fig = plt.figure(figsize=(10, 6))

# création de l'histogramme
user_interactions.hist(bins=100)


plt.title('Nombre d\'évaluations par utilisateur')
plt.xlabel('Nombre d\'évaluations')
plt.ylabel('Nombre d\'utilisateurs')
plt.show()

# Calcul du nombre d'évaluations par item
item_interactions = ratings_df.groupby('movie_id')['rating'].count()

# création de la figure
fig = plt.figure(figsize=(10, 6))

# création de l'histogramme
item_interactions.hist(bins=100)

# étiquettes et titre
plt.title('Nombre d\'évaluations par item')
plt.xlabel('Nombre d\'évaluations')
plt.ylabel('Nombre d\'items')
plt.show()


# Répartition des notes
fig = plt.figure(figsize=(10, 6))
ratings_df['rating'].hist(bins=[0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
plt.title('Répartition des notes')
plt.xlabel('Note')
plt.ylabel('Nombre d\'évaluations')
plt.show()


