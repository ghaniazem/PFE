import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split


# Load the MovieLens 1M dataset
ratings_df = pd.read_csv('dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
movies_df = pd.read_csv('dataset/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python')
users_df = pd.read_csv("dataset/users.dat", sep="::", names=["user_id", "gender", "age", "occupation", "zipcode"], engine='python')

#print(movies_df.head())
#------movies------
# Encodage de la colonne "movie_id"
movies_df["movie_id"] = pd.Categorical(movies_df["movie_id"]).codes

# Encodage de la colonne "title"
movies_df["title"] = pd.Categorical(movies_df["title"]).codes

# Encodage de la colonne "genres"
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies_df["genres"].str.split("|"))
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
movies_df = pd.concat([movies_df, genres_df], axis=1).drop("genres", axis=1)

#------ratings-------
# Encodage de la colonne "user_id"
ratings_df["user_id"] = pd.Categorical(ratings_df["user_id"]).codes

# Encodage de la colonne "movie_id"
ratings_df["movie_id"] = pd.Categorical(ratings_df["movie_id"]).codes

#---------users---------
# Encodage de la colonne "gender"
le = LabelEncoder()
users_df["gender"] = le.fit_transform(users_df["gender"])

# Encodage de la colonne "user_id"
users_df["user_id"] = le.fit_transform(users_df["user_id"])

#fusionner les fichiers
ratings_movies = pd.merge(ratings_df, movies_df, on='movie_id')

#suppression des colonnes inutiles
ratings_movies.drop(['timestamp'], axis=1, inplace=True)
#print(ratings_movies)

# Supprimer les lignes en double
ratings_movies = ratings_movies.drop_duplicates()
#print(ratings_movies)


"""# Afficher un boxplot de la colonne "rating"
plt.boxplot(ratings_movies['rating'])

# Ajouter un titre et des labels d'axes
plt.title('Boxplot de la colonne "rating"')
plt.xlabel('rating')
plt.show()"""

# Calculer la moyenne et l'écart type de la colonne "rating"
mean_rating = ratings_movies["rating"].mean()
std_rating = ratings_movies["rating"].std()

# Calculer le score Z de chaque valeur
ratings_movies["z_score"] = np.abs(stats.zscore(ratings_movies["rating"]))

# Définir un seuil de score Z pour supprimer les valeurs aberrantes
threshold = 2.3

# Créer une nouvelle DataFrame contenant uniquement les valeurs non aberrantes
ratings_movies = ratings_movies[ratings_movies["z_score"] < threshold]

# Supprimer la colonne "z_score"
ratings_movies.drop(["z_score"], axis=1, inplace=True)

"""# Afficher un boxplot de la colonne "rating"
plt.boxplot(ratings_movies['rating'])

# Ajouter un titre et des labels d'axes
plt.title('Boxplot de la colonne "rating"')
plt.xlabel('rating')
plt.show()"""

#__________________________________________________________
# Calculer la moyenne de chaque colonne
mean_ratings = ratings_df.mean(axis=0)

# Remplacer les valeurs manquantes par la moyenne de chaque colonne
ratings_movies.fillna(mean_ratings, inplace=True)

# Vérifier si les données manquantes ont été supprimées
print(ratings_movies.isna().sum())

#convertir le data frame en tabelau
df_array = ratings_movies.to_numpy()

#mélanger les données
np.random.shuffle(df_array)
print(df_array)

# Diviser les données en ensembles d'entraînement et de test (random_state pour que les résultats soient reproductibles)
train_data, test_data = train_test_split(df_array, test_size=0.2, random_state=42)
#print("Taille de train_data :", np.shape(train_data))
#print("Taille de test_data :", np.shape(test_data))