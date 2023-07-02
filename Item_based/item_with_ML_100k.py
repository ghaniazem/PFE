# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score, precision_score, recall_score


# Load the MovieLens 1M dataset
ratings_movies = pd.read_csv('../ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

#------ratings-------
# Encodage de la colonne "user_id"
ratings_movies["user_id"] = pd.Categorical(ratings_movies["user_id"]).codes

# Encodage de la colonne "movie_id"
ratings_movies["movie_id"] = pd.Categorical(ratings_movies["movie_id"]).codes


#suppression des colonnes inutiles
ratings_movies.drop(['timestamp'], axis=1, inplace=True)

movie = np.unique(ratings_movies['movie_id'])
#print("le nombre d'user dans ratings_movies : ",len(movie))


train_df, test_df = train_test_split(ratings_movies, test_size=0.2, random_state=42)

movie_train = np.unique(train_df['movie_id'])
#print("le nombre d'user dans train_movies : ",len(movie_train))

num_users = train_df['user_id'].nunique()
num_movies = train_df['movie_id'].nunique()

# Créer une matrice d'utilité
utility_matrix = train_df.pivot_table(values='rating', index='user_id', columns='movie_id')
utility_matrix = utility_matrix.fillna(0)

train_movies = list(utility_matrix.columns)
movies_in_training = train_df['movie_id'].unique()

def computeAdjCosSim(ratings_matrix):
    # Calcul de la matrice de similarité cosinus ajustée
    item_means = np.mean(ratings_matrix, axis=0)  # Moyenne de chaque item (film)
    item_adjusted = ratings_matrix - item_means  # Notations ajustées pour chaque item
    similarity_matrix = np.dot(item_adjusted.T, item_adjusted) / (np.linalg.norm(item_adjusted.T, axis=1, keepdims=True) * np.linalg.norm(item_adjusted, axis=0))

    return similarity_matrix

item_similarity = computeAdjCosSim(utility_matrix)


def predict(user_id, item_id, ratings_matrix, similarity_matrix, k):
    # Obtenez les similarités pour l'article cible i
    similarities = similarity_matrix[item_id, :]

    # Triez les similarités de manière décroissante et obtenez les indices correspondants
    similar_indices = np.argsort(similarities)[::-1]

    # Sélectionnez les k indices des articles les plus similaires (à l'exception de l'article cible lui-même)
    top_k_indices = similar_indices[1:k + 1]

    #récupérer les films notées par l'utilisateur
    positive_ratings_indices = [i for i in range(len(ratings_matrix[user_id])) if ratings_matrix[user_id][i] > 0]

    articles_achetes_dans_similaires = []
    for article in positive_ratings_indices:
        if article in top_k_indices:
            articles_achetes_dans_similaires.append(article)

    if not articles_achetes_dans_similaires:
        # Aucun film noté par l'utilisateur parmi les films similaires
        return 0

    #récupérer les notes des films de l'utilisateur
    user_ratings = ratings_matrix[user_id, :]
    similar_rating = user_ratings[articles_achetes_dans_similaires]

    #récupérer les similarité des films similaires
    similar_similariy = similarity_matrix[articles_achetes_dans_similaires]

    #calculer les note prédite par la somme pondérée
    weight = np.dot(similar_rating, similar_similariy)
    predicted_rating = np.sum(weight) / np.sum(similar_similariy)

    return predicted_rating



relvent_movies_list = []
recommandations = []
for user_id in test_df['user_id'].unique():
    # Obtenir les films notés par cet utilisateur dans l'ensemble de test
    rated_movies = test_df[test_df['user_id'] == user_id]['movie_id'].values
    # Créer une liste pour stocker les scores de prédiction pour chaque film
    user_predictions = []
    # Parcourir chaque film dans l'ensemble de test
    for movie_id in test_df[test_df['user_id'] == user_id]['movie_id'].values:
        # Vérifier si le film est dans l'ensemble d'entraînement et a une similarité calculée
        if movie_id in movies_in_training and movie_id < item_similarity.shape[0]:
            # Prédire la note pour ce film
            predicted_rating = predict(user_id, movie_id, utility_matrix.values, item_similarity, 200)
            # Ajouter le score de prédiction à la liste (liste de films prédits avec leur notes pour chaque utilisateur )
            user_predictions.append((movie_id, predicted_rating))


    # Trier les scores de prédiction par ordre décroissant
    user_predictions.sort(key=lambda x: x[1], reverse=True)
    # Obtenir les recommandations (les films avec les scores les plus élevés)
    user_recommendations = [movie_id for movie_id, predicted_rating in user_predictions if predicted_rating > 0]
    #user_recommendations = [movie_id for movie_id, _ in user_predictions]
    print(user_recommendations)
    # Ajouter les recommandations à la liste des recommandations globales
    recommandations.append((user_id, user_recommendations))
    relvent_movies_list.append((user_id, rated_movies))
    #print(recommandations, "\n")

# Calcul des scores de rappel, nDCG et precision pour différentes valeurs de n
recall_scores = []
ndcg_scores = []
precision_scores = []
for n in range(5, 225, 25):
    recall_at_n = []
    ndcg_at_n = []
    precision_at_n = []

    for user_id, user_recommendations in recommandations:
        # Obtenir les films réellement notés par cet utilisateur dans l'ensemble de test
        actual_movies = test_df[test_df['user_id'] == user_id]['movie_id'].values

        # Générer les étiquettes binaires pour les films recommandés et les films réels
        y_true = [1 if movie_id in actual_movies else 0 for movie_id in movie]
        y_pred = [1 if movie_id in user_recommendations[:n] else 0 for movie_id in movie]

        # Calculer le score nDCG en utilisant la fonction ndcg_score
        ndcg = ndcg_score([y_true], [y_pred], k=n)
        ndcg_at_n.append(ndcg)

        #calculer le rappel
        recall = recall_score(y_true, y_pred)
        recall_at_n.append(recall)

        # Calculer la précision pour cette valeur de n
        precision = precision_score(y_true, y_pred, zero_division=1)
        precision_at_n.append(precision)

    # Calculer la moyenne des scores de rappel pour cette valeur de n
    mean_recall = np.mean(recall_at_n)
    recall_scores.append((n, mean_recall))

    # Calculer la moyenne des scores nDCG pour cette valeur de n
    mean_ndcg = np.mean(ndcg_at_n)
    ndcg_scores.append((n, mean_ndcg))

    # Calculer la moyenne des scores de précision pour cette valeur de n
    mean_precision = np.mean(precision_at_n)
    precision_scores.append((n, mean_precision))

for n, recall in recall_scores:
    print("Rappel@{} : {:.2f}".format(n, recall))

for n, ndcg in ndcg_scores:
    print("NDCG@{} : {:.2f}".format(n, ndcg))


# Extraire les valeurs de n et les scores de rappel
n_values = [n for n, _ in recall_scores]
recall_values = [recall for _, recall in recall_scores]

# Extraire les scores nDCG
ndcg_values = [ndcg for _, ndcg in ndcg_scores]

# Extraire les valeurs de n et les scores de précision
#n_values = [n for n, _ in precision_scores]
precision_values = [precision for _, precision in precision_scores]


# Tracer le graphe du rappel et nDCG
plt.plot(n_values, recall_values, marker='o', label='Rappel')
plt.plot(n_values, ndcg_values, marker='o', label='NDCG')
plt.plot(n_values, precision_values, marker='o', label='Precision')
plt.xlabel('Valeur de K')
plt.ylabel('Score')
plt.title('Rappel, NDCG et Precision en fonction de nombre de recommandations')
plt.legend()
plt.grid(True)
plt.show()

# Calcul des scores de rappel pour différentes valeurs de n
recall_scores = []
for user_id, user_recommendations in recommandations:
    # Obtenir les films réellement notés par cet utilisateur dans l'ensemble de test
    actual_movies = test_df[test_df['user_id'] == user_id]['movie_id'].values
    # Générer les étiquettes binaires pour les films recommandés et les films réels
    y_true = [1 if movie_id in actual_movies else 0 for movie_id in movie]
    y_pred = [1 if movie_id in user_recommendations else 0 for movie_id in movie]
    # Calculer le rappel en utilisant la fonction recall_score
    recall = recall_score(y_true, y_pred)
    recall_scores.append(recall)

# Calculer la moyenne des scores de rappel
mean_recall = np.mean(recall_scores)
print(mean_recall)

recall_scores_ = []

for n in range(5, 11, 5):
    recall_at_n = []
    for user_id, user_recommendations in recommandations:
        # Obtenir les films réellement notés par cet utilisateur dans l'ensemble de test
        actual_movies = test_df[test_df['user_id'] == user_id]['movie_id'].values
        # Générer les étiquettes binaires pour les films recommandés et les films réels
        y_true = [1 if movie_id in actual_movies else 0 for movie_id in movie]
        y_pred = [1 if movie_id in user_recommendations[:n] else 0 for movie_id in movie]
        # Calculer le rappel en utilisant la fonction recall_score
        recall = recall_score(y_true, y_pred)
        recall_at_n.append(recall)

    # Calculer la moyenne des scores de rappel pour cette valeur de n
    mean_recall_ = np.mean(recall_at_n)
    recall_scores_.append((n, mean_recall_))

# Afficher les scores de rappel pour chaque valeur de n
for n, recall in recall_scores_:
    print("Rappel@{} : {:.2f}".format(n, recall))

# Extraire les valeurs de n et les scores de rappel
n_values = [n for n, _ in recall_scores_]
recall_values = [recall for _, recall in recall_scores_]

# Tracer le graphe
plt.plot(n_values, recall_values, marker='o')
plt.xlabel('Valeur de n')
plt.ylabel('Rappel')
plt.title('Rappel en fonction de la valeur de n')
plt.grid(True)
plt.show()

# Créer le diagramme à barres
plt.bar(n_values, recall_values, width=1)
plt.xlabel('Valeur de k')
plt.ylabel('Valeur de rappel')
plt.title('Rappel@k')
plt.xlim(1, 11)
plt.xticks(n_values)
plt.show()

# Calcul des scores de nDCG pour différentes valeurs de n
ndcg_scores = []

for n in range(5, 11, 5):
    ndcg_at_n = []
    for user_id, user_recommendations in recommandations:
        # Obtenir les films réellement notés par cet utilisateur dans l'ensemble de test
        actual_movies = test_df[test_df['user_id'] == user_id]['movie_id'].values
        # Générer les étiquettes binaires pour les films recommandés et les films réels
        y_true = [1 if movie_id in actual_movies else 0 for movie_id in movie]
        y_pred = [1 if movie_id in user_recommendations[:n] else 0 for movie_id in movie]
        # Calculer le score nDCG en utilisant la fonction ndcg_score
        ndcg = ndcg_score([y_true], [y_pred], k=n)
        ndcg_at_n.append(ndcg)

    # Calculer la moyenne des scores nDCG pour cette valeur de n
    mean_ndcg = np.mean(ndcg_at_n)
    ndcg_scores.append((n, mean_ndcg))

# Extraire les valeurs de n et les scores nDCG
n_values = [n for n, _ in ndcg_scores]
ndcg_values = [ndcg for _, ndcg in ndcg_scores]

# Tracer le graphe
plt.plot(n_values, ndcg_values, marker='o')
plt.xlabel('Valeur de n')
plt.ylabel('nDCG')
plt.title('nDCG en fonction de la valeur de n')
plt.grid(True)
plt.show()

# Créer le diagramme à barres
plt.bar(n_values, ndcg_values, width=1)
plt.xlabel('Valeur de k')
plt.ylabel('Valeur de NDCG')
plt.title('NDCG@k')
plt.xlim(1, 11)
plt.xticks(n_values)
plt.show()

# Calcul des scores de précision pour différentes valeurs de n
precision_scores = []

for n in range(5, 11, 5):
    precision_at_n = []

    for user_id, user_recommendations in recommandations:
        # Obtenir les films réellement notés par cet utilisateur dans l'ensemble de test
        actual_movies = test_df[test_df['user_id'] == user_id]['movie_id'].values

        # Générer les étiquettes binaires pour les films recommandés et les films réels
        y_true = [1 if movie_id in actual_movies else 0 for movie_id in movie]
        y_pred = [1 if movie_id in user_recommendations[:n] else 0 for movie_id in movie]

        # Calculer la précision pour cette valeur de n
        precision = precision_score(y_true, y_pred, zero_division=1)
        precision_at_n.append(precision)

    # Calculer la moyenne des scores de précision pour cette valeur de n
    mean_precision = np.mean(precision_at_n)
    precision_scores.append((n, mean_precision))

# Extraire les valeurs de n et les scores de précision
n_values = [n for n, _ in precision_scores]
precision_values = [precision for _, precision in precision_scores]

# Tracer le graphe de la précision
plt.plot(n_values, precision_values, marker='o')
plt.xlabel('Valeur de n')
plt.ylabel('Précision')
plt.title('Précision en fonction de la valeur de n')
plt.grid(True)
plt.show()

# Créer le diagramme à barres
plt.bar(n_values, precision_values, width=1)
plt.xlabel('Valeur de k')
plt.ylabel('Valeur de précision')
plt.title('Précision@k')
plt.xlim(1, 11)
plt.xticks(n_values)
plt.show()


