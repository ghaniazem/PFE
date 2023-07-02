import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

# Charger les données
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

# Division des données en ensemble de formation et ensemble de test
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Initialisation des matrices de facteurs latents pour les utilisateurs et les éléments
latent_dim = 108
user_latent_matrix = np.random.randn(num_users, latent_dim)
item_latent_matrix = np.random.randn(num_items, latent_dim)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Hyperparamètres
num_epochs = 101
learning_rate = 0.01
lambda_reg = 0.001
k_values = [5, 10]

# Dictionnaire pour stocker les valeurs de rappel à k
recall_values = {k: [] for k in k_values}

# Boucle d'entraînement
for epoch in range(num_epochs):
    # Mélanger les données d'entraînement
    train_data = train_data.sample(frac=1)

    for _, row in train_data.iterrows():
        user = row['user_id']
        item = row['movie_id']
        rating = row['rating']

        # Paire positive
        if rating >= 1.0:
            positive_score = np.dot(user_latent_matrix[user], item_latent_matrix[item])
            negative_item = np.random.choice(np.setdiff1d(np.arange(num_items), item))
            negative_score = np.dot(user_latent_matrix[user], item_latent_matrix[negative_item])
            score_diff = positive_score - negative_score

            # Mise à jour des matrices de facteurs latents avec la descente de gradient et la régularisation
            user_latent_matrix[user] += learning_rate * ((item_latent_matrix[item] - item_latent_matrix[negative_item]) * sigmoid(-score_diff) - lambda_reg * user_latent_matrix[user])
            item_latent_matrix[item] += learning_rate * (user_latent_matrix[user] * sigmoid(-score_diff) - lambda_reg * item_latent_matrix[item])
            item_latent_matrix[negative_item] += learning_rate * ((-user_latent_matrix[user]) * sigmoid(-score_diff) - lambda_reg * item_latent_matrix[negative_item])

    # Fonction de prédiction pour obtenir les recommandations top-k (sans les titres des films)
    def predict(user_id, top_k=10):
        user_vector = user_latent_matrix[user_id]
        scores = np.dot(item_latent_matrix, user_vector)
        top_items = np.argsort(scores)[::-1][:top_k]
        recommended_movies = list(top_items)
        return recommended_movies

    # Calculer le rappel à k pour chaque utilisateur
    for k in k_values:
        recall_at_k_list = []
        for user_id in test_data['user_id'].unique():
            user_ratings = test_data[test_data['user_id'] == user_id]['movie_id'].values
            recommended_movies = predict(user_id, top_k=k)
            labels = [1 if i in user_ratings else 0 for i in np.arange(num_items)]
            recall = recall_score(labels, [1 if i in recommended_movies else 0 for i in np.arange(num_items)])
            recall_at_k_list.append(recall)

        # Moyenne du rappel à k sur tous les utilisateurs
        mean_recall_at_k = np.mean(recall_at_k_list)
        recall_values[k].append(mean_recall_at_k)

    print("Epoch:", epoch + 1, "completed")


# Affichage des résultats
for k in k_values:
    plt.plot(range(num_epochs), recall_values[k], label=f'k={k}')


# Spécifier le pas d'itérations
step_size = 5  # Pas d'itérations souhaité
plt.xticks(range(0, num_epochs, step_size))

plt.xlabel('Nombre d\'epochs')
plt.ylabel('Rappel@k')
plt.title('Rappel@k en fonction du nombre d\'epochs')
plt.legend()

plt.show()