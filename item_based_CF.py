import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


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


# Vérifier si les données manquantes ont été supprimées
#print(ratings_movies.isna().sum())
#print(ratings_movies[ratings_movies.isna().any(axis=1)])

#convertir le data frame en tabelau
df_array = ratings_movies.to_numpy()

#mélanger les données
#np.random.shuffle(df_array)
#print(df_array)

# Diviser les données en ensembles d'entraînement et de test (random_state pour que les résultats soient reproductibles)
train_data, test_data = train_test_split(df_array, test_size=0.2, random_state=42)
#print("Taille de train_data :", np.shape(train_data))
#print("Taille de test_data :", np.shape(test_data))
#print(test_data)


# Créer une liste de tous les utilisateurs uniques dans train_data
users = np.unique(train_data[:, 0])

# Créer une liste de tous les films uniques dans train_data
movies = np.unique(train_data[:, 1])

# Créer une matrice numpy de forme (nombre d'utilisateurs, nombre de films) remplie de zéros.
utility_matrix = np.zeros((len(users), len(movies)))


# Parcourir chaque élément de train_data et mettre à jour la valeur de la matrice à l'indice correspondant pour chaque utilisateur et film avec la note correspondante.
for rating in train_data:
    user_idx = np.where(users == rating[0])[0][0]
    movie_idx = np.where(movies == rating[1])[0][0]
    utility_matrix[user_idx, movie_idx] = rating[2]

"""for j in range(utility_matrix.shape[1]):
    col = utility_matrix[:, j]
    col_mean = np.mean(col[col!=0]) # calculer la moyenne des valeurs non nulles
    col[col==0] = col_mean # remplacer les valeurs nulles par la moyenne
    utility_matrix[:, j] = col"""

#print(utility_matrix)

# Calcul de la matrice de similarité article-article
item_similarity = cosine_similarity(utility_matrix.T)
#item_similarity = item_similarity.T
#print(item_similarity)
#n_users, n_items = item_similarity.shape
#print(n_users, n_items)

# Define the number of similar users to use for prediction
k = 5

#la fonction predict
def predict(user_id, item_id, ratings_matrix, similarity_matrix):
    #récupérer les indices des films notée par l'utilisateur
    #positive_ratings_indices = [i for i in range(len(ratings_matrix[user_id])) if ratings_matrix[user_id][i] > 0]


    #racupérer les similarités entre les films notées par l'utilisateur
    #similarities = similarity_matrix[positive_ratings_indices, :]

    #Extraire les similarités entre le film cible et les autres films notés par l'user
    similarities = similarity_matrix[item_id, :]

    #Sélectionner les indices des films ayant une similarité supérieure ou égale à 0.3
    similar_movies_indices = [i for i, similarity in enumerate(similarities) if
                              similarity >= 0.4 and i != item_id]

    #récupérer les notes des filmes similaires
    user_ratings = ratings_matrix[user_id, :]
    similar_ratings = user_ratings[similar_movies_indices]
    similar_ratings_list = similar_ratings.tolist()

    #récupérer les similarités entre le film cible et les autres films les plus similaires
    target_movie_similarity = item_similarity[item_id, :]
    similar_movie_similarity = target_movie_similarity[similar_movies_indices]
    similar_movie_similarity_list = similar_movie_similarity.tolist()

    #calculer la note prédite
    weighted_ratings = similar_ratings_list * similar_movie_similarity
    weighted_sum = weighted_ratings.sum()
    similarity_sum = similar_movie_similarity.sum()
    if similarity_sum > 0 and not np.isnan(similarity_sum):
        predicted_rating = weighted_sum / similarity_sum
    else:
        predicted_rating = 0 # or any other value that makes sense in your context

    return predicted_rating

#predict(745,1,utility_matrix, item_similarity)
for row in test_data:
    user_id = row[0]
    movie_id = row[1]
    true_rating = row[2]
    predicted_rating = predict(user_id, movie_id, utility_matrix, item_similarity)
    print(predicted_rating)


