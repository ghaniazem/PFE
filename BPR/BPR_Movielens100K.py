# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, ndcg_score, precision_score
import matplotlib.pyplot as plt

# Charger les données du dataset 1M
#ratings = pd.read_csv('../dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')


# Charger les données du dataset 100K
ratings = pd.read_csv('../ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Prétraitement des données
# Supprimer les colonnes inutiles
ratings = ratings.drop(['timestamp'], axis=1)

# Convertir les identifiants en indices numériques
user_mapping = {user_id: i for i, user_id in enumerate(ratings['user_id'].unique())}
item_mapping = {item_id: i for i, item_id in enumerate(ratings['movie_id'].unique())}
ratings['user_id'] = ratings['user_id'].map(user_mapping)
ratings['movie_id'] = ratings['movie_id'].map(item_mapping)

# Obtention du nombre d'utilisateurs et d'éléments
num_users = ratings['user_id'].nunique()
num_items = ratings['movie_id'].nunique()
print("num_users:", num_users)
print("num_items:", num_items)

# Division des données en ensemble de formation et ensemble de test
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Initialisation des matrices de facteurs latents pour les utilisateurs et les éléments
latent_dim = 100
user_latent_matrix = np.random.randn(num_users, latent_dim)
item_latent_matrix = np.random.randn(num_items, latent_dim)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperparamètres
learning_rate = 0.01
k = 100
lambda_reg = 0.001

# Listes pour stocker les scores
precision_scores = []
recall_scores = []
ndcg_scores = []
epochs = range(0, 61, 5)

# Boucle d'entraînement
for num_epochs in epochs:
    for epoch in range(num_epochs):
        # Mélanger les données d'entraînement
        train_data = train_data.sample(frac=1)

        for _, row in train_data.iterrows():
            user = row['user_id']
            item = row['movie_id']
            rating = row['rating']

            # Paire positive
            if rating >= 2.0:
                positive_score = np.dot(user_latent_matrix[user], item_latent_matrix[item])
                negative_item = np.random.choice(np.setdiff1d(np.arange(num_items), item))
                negative_score = np.dot(user_latent_matrix[user], item_latent_matrix[negative_item])
                score_diff = positive_score - negative_score

                # Mise à jour des matrices de facteurs latents avec la descente de gradient et la régularisation
                user_latent_matrix[user] += learning_rate * ((item_latent_matrix[item] - item_latent_matrix[negative_item]) * sigmoid(-score_diff) - lambda_reg * user_latent_matrix[user])
                item_latent_matrix[item] += learning_rate * (user_latent_matrix[user] * sigmoid(-score_diff) - lambda_reg * item_latent_matrix[item])
                item_latent_matrix[negative_item] += learning_rate * ((-user_latent_matrix[user]) * sigmoid(-score_diff) - lambda_reg * item_latent_matrix[negative_item])


    print("Epoch:", num_epochs, "completed")



    # Évaluation du modèle avec le score de précision
    y_true = test_data['rating'] >= 2.0
    y_pred = np.array([np.dot(user_latent_matrix[user], item_latent_matrix[item]) for user, item in zip(test_data['user_id'], test_data['movie_id'])]) >= 2.0

    precision = precision_score(y_true, y_pred)
    precision_scores.append(precision)

    # Évaluation du modèle avec le score de rappel
    recall = recall_score(y_true, y_pred)
    recall_scores.append(recall)

    # Évaluation du modèle avec le score NDCG
    test_user_ids = test_data['user_id'].map(user_mapping).values
    test_item_ids = test_data['movie_id'].map(item_mapping).values
    test_user_ids = test_user_ids.astype(int)
    test_item_ids = test_item_ids.astype(int)

    predicted_ratings = []
    for user, item in zip(test_user_ids, test_item_ids):
        if 0 <= user < num_users and 0 <= item < num_items:
            rating = np.dot(user_latent_matrix[user], item_latent_matrix[item])
            predicted_ratings.append(rating)
        else:
            predicted_ratings.append(0.0)

    predicted_ratings = np.array(predicted_ratings)
    ndcg = ndcg_score(test_data['rating'].values.reshape(1, -1), predicted_ratings.reshape(1, -1))
    ndcg_scores.append(ndcg)

# Affichage des graphiques
plt.plot(epochs, precision_scores, label='Précision')
plt.plot(epochs, recall_scores, label='Rappel')
plt.plot(epochs, ndcg_scores, label='NDCG')
plt.xlabel('Nombre d\'epochs')
plt.ylabel('Score')
plt.legend()
plt.show()