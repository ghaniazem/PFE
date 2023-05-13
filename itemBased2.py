import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix


# Load the MovieLens 1M dataset
ratings_df = pd.read_csv('dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
movies_df = pd.read_csv('dataset/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python')
users_df = pd.read_csv("dataset/users.dat", sep="::", names=["user_id", "gender", "age", "occupation", "zipcode"], engine='python')

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
ratings_movies = ratings_movies.fillna(mean_ratings)
#print(ratings_movies.isna().sum())

train_df, test_df = train_test_split(ratings_movies, test_size=0.2, random_state=42)

# Créer une matrice d'utilité
utility_matrix = train_df.pivot_table(values='rating', index='user_id', columns='movie_id')
#print(utility_matrix)

item_similarity = cosine_similarity(utility_matrix.fillna(0))
print(item_similarity)

"""def predict(user_id, movie_id, utility_matrix, item_similarity):
  Fonction pour prédire la note d'un utilisateur pour un film donné en utilisant la similarité cosinus
    user_ratings = utility_matrix.loc[user_id].fillna(0).values.reshape(1, -1)
    similarities = item_similarity[movie_id].reshape(1, -1)
    prediction = user_ratings.dot(similarities)/np.sum(similarities)
    return prediction[0, 0]

# Prédire les notes des utilisateurs pour les films du jeu de test
predictions = []
for _, row in test_df.iterrows():
    user_id = row["user_id"]
    movie_id = row["movie_id"]
    rating = row["rating"]
    predicted_rating = predict(user_id, movie_id, utility_matrix, item_similarity)
    predictions.append((user_id, movie_id, rating, predicted_rating))

print(predictions)"""


"""print(utility_matrix.shape)

def predict(user_id, movie_id, utility_matrix, item_similarity, k=10):
    Fonction pour prédire la note d'un utilisateur pour un film donné en utilisant les k plus proches voisins
    # Trouver les indices des k plus proches voisins
    neigh = NearestNeighbors(n_neighbors=k, metric='cosine')
    neigh.fit(item_similarity)
    most_similar = neigh.kneighbors(X=item_similarity[movie_id].reshape(1, -1), return_distance=False)

    # Extraire les notes des utilisateurs pour les films similaires
    similar_ratings = utility_matrix.iloc[user_id, most_similar[0]].values

    # Calculer la prédiction en utilisant les moyens pondérés
    similarities = item_similarity[movie_id, most_similar[0]]
    prediction = np.sum(similar_ratings * similarities) / np.sum(similarities)

    return prediction


predictions = []
# Itérer sur chaque ligne de l'ensemble de test
for index, row in test_df.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']

    # Faire une prédiction pour l'utilisateur et le film donnés
    predicted_rating = predict(user_id, movie_id, utility_matrix, item_similarity)

    # Ajouter la prédiction à une liste
    predictions.append(predicted_rating)

# Convertir la liste de prédictions en un tableau numpy
predictions = np.array(predictions)

# Afficher les prédictions
print(predictions)"""


"""utility_matrix.fillna(0, inplace=True)

# Conversion en une matrice creuse CSR pour une utilisation plus efficace avec Scipy
utility_csr = csr_matrix(utility_matrix.values)


# Calcul de la similarité cosine entre les items
item_similarity = cosine_similarity(utility_csr.T)
"""

