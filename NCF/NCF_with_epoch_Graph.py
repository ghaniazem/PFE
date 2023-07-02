import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Multiply, Concatenate
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import recall_score, ndcg_score, precision_score
from keras.optimizers import Adam
from keras.layers import RepeatVector, Reshape
from keras.regularizers import l2
import matplotlib.pyplot as plt

# Load the MovieLens 1M dataset
#df = pd.read_csv('../dataset/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
# Load the MovieLens 100K dataset
df = pd.read_csv('../ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
ratings_df = df[df['rating'].isin([4, 5])]

# Encodage de la colonne "user_id"
#ratings_df["user_id"] = pd.Categorical(ratings_df["user_id"]).codes
ratings_df.loc[:, "user_id"] = pd.Categorical(ratings_df["user_id"]).codes
# Encodage de la colonne "movie_id"
#ratings_df["movie_id"] = pd.Categorical(ratings_df["movie_id"]).codes
ratings_df.loc[:, "movie_id"] = pd.Categorical(ratings_df["movie_id"]).codes

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

# Déterminez les valeurs maximale et minimale de la colonne "rating"
max_rating = train_data['rating'].max()
min_rating = df['rating'].min()

# Créer la matrice utilisateur-item binaire
user_ids = train_data['user_id'].unique()
movie_ids = train_data['movie_id'].unique()

train_movie_ids = set(train_data['movie_id'])
test_movie_ids = set(test_data['movie_id'])

max_movie_id = train_data['movie_id'].max()

matrix = pd.DataFrame(0, index=user_ids, columns=range(1, max_movie_id + 1))

for _, row in train_data.iterrows():
    user_id = row['user_id']
    movie_id = row['movie_id']
    rating = row['rating']
    if rating >= 1 :
        matrix.loc[user_id, movie_id] = 1


# Appliquez la normalisation min-max à la colonne "rating"
train_data['rating'] = (train_data['rating'] - min_rating) / (max_rating - min_rating)

# Paramètres
lr = 0.001  # Taux d'apprentissage
#latent_dims = [350]  # Valeurs de la dimension latente à boucler
latent_dim = 350
#layers = [64, 32, 16, 8]  # Dimensions des couches du MLP
epochs = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # Valeurs du nombre d'epochs à boucler
#epoch = 400
recall_values = []  # Exemple de valeurs de rappel
precision_values = []  # Exemple de valeurs de précision
ndcg_values = []

for epoch in epochs:
    """---------------GMF--------------"""
    # Entrées du modèle
    user_input_gmf = Input(shape=(1,))
    item_input_gmf = Input(shape=(1,))

    # Embedding des utilisateurs et des films
    user_embedding_gmf = Embedding(input_dim=len(user_ids), output_dim=latent_dim, name='user_embedding')(user_input_gmf)
    item_embedding_gmf = Embedding(input_dim=max_movie_id + 1, output_dim=latent_dim, name='item_embedding')(item_input_gmf)

    # Applatissement des vecteurs d'embedding (rendre les matrices multi-dim des embeddings en un vecteur uni-dim)
    user_flat = Flatten()(user_embedding_gmf)
    item_flat = Flatten()(item_embedding_gmf)

    # Element-wise product des vecteurs d'embedding
    merged = Multiply()([user_flat, item_flat])

    # Couche de sortie avec activation sigmoid
    output_gmf = Dense(1, activation='sigmoid', name='prediction')(merged)

    # Création du modèle GMF
    model = Model(inputs=[user_input_gmf, item_input_gmf], outputs=output_gmf)

    # Compilation du modèle
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy')

    # Entraînement du modèle
    #early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    model.fit([train_data['user_id'], train_data['movie_id']], train_data['rating'], epochs=10, batch_size=512, validation_split=0.2)

    # Faire des prédictions avec le modèle GMF
    user_ids_test = test_data['user_id'].values
    movie_ids_test = test_data['movie_id'].values

    # -------------MLP------------
    # Définir les dimensions des vecteurs d'embedding pour la partie MLP
    l2_regularization = 0.001  # Force de la régularisation L2
    # Entrées du modèle
    user_input_mlp = Input(shape=(1,))
    item_input_mlp = Input(shape=(1,))

    # Embedding des utilisateurs et des films pour la partie MLP
    user_embedding_mlp = Embedding(input_dim=matrix.shape[0], output_dim=latent_dim, name='user_embedding')(
        user_input_mlp)
    item_embedding_mlp = Embedding(input_dim=matrix.shape[1], output_dim=latent_dim, name='item_embedding')(
        item_input_mlp)

    # Applatissement des vecteurs d'embedding pour la partie MLP
    user_flat_mlp = Flatten()(user_embedding_mlp)
    item_flat_mlp = Flatten()(item_embedding_mlp)

    # Define the MLP layers
    hidden_units = [32, 16, 8]  # Number of neurons in each hidden layer
    activations = ['relu', 'relu', 'relu']  # Activation functions for each hidden layer

    # Concaténation des vecteurs d'embedding pour la partie MLP
    concatenated_mlp = Concatenate()([user_flat_mlp, item_flat_mlp])

    # MLP layers
    mlp_output = concatenated_mlp
    for units, activation in zip(hidden_units, activations):
        mlp_output = Dense(units, activation=activation)(mlp_output)

    # Output layer
    mlp_output = Dense(1, activation='sigmoid', name='prediction', kernel_regularizer=l2(l2_regularization),
                       bias_regularizer=l2(l2_regularization))(mlp_output)

    # Create the MLP model
    mlp_model = Model(inputs=[user_input_mlp, item_input_mlp], outputs=mlp_output)

    # Compilation du modèle
    mlp_model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy')

    # Entraînement du modèle
    # early_stopping = EarlyStopping(patience=3, restore_best_weights=True)
    mlp_model.fit([train_data['user_id'], train_data['movie_id']], train_data['rating'], epochs=10, batch_size=512,
                  validation_split=0.2)

    # --------------NeuMF---------------

    # Déclaration des tensors pour user_input_neumf et item_input_neumf
    user_input_neumf = tf.keras.Input(shape=(1,), name='user_input_neumf')
    item_input_neumf = tf.keras.Input(shape=(1,), name='item_input_neumf')
    # Création du modèle NeuMF
    neumf_user_embedding = Embedding(input_dim=matrix.shape[0], output_dim=latent_dim, name='mf_embedding_user')(
        user_input_neumf)
    neumf_item_embedding = Embedding(input_dim=matrix.shape[1], output_dim=latent_dim, name='mf_embedding_item')(
        item_input_neumf)

    neumf_gmf_output = Multiply()([neumf_user_embedding, neumf_item_embedding])

    neumf_mlp_user_embedding = Embedding(input_dim=matrix.shape[0], output_dim=latent_dim, name='mlp_embedding_user')(
        user_input_neumf)
    neumf_mlp_item_embedding = Embedding(input_dim=matrix.shape[1], output_dim=latent_dim, name='mlp_embedding_item')(
        item_input_neumf)

    neumf_mlp_output = Concatenate()([neumf_mlp_user_embedding, neumf_mlp_item_embedding])
    neumf_mlp_output = Flatten()(neumf_mlp_output)

    for units, activation in zip(hidden_units, activations):
        neumf_mlp_output = Dense(units, activation=activation)(neumf_mlp_output)

    print(neumf_mlp_output.shape)
    print(neumf_gmf_output.shape)

    gmf_output_flattened = Flatten()(neumf_gmf_output)  # Ajuster la forme du tenseur (None, 1, 24) en (None, 24)

    # Concaténation des sorties GMF et MLP
    neumf_output = Concatenate()([gmf_output_flattened, neumf_mlp_output])
    neumf_output = Dense(1, activation='sigmoid', name='prediction')(neumf_output)

    # Création du modèle NeuMF
    neumf_model = Model(inputs=[user_input_neumf, item_input_neumf], outputs=neumf_output)

    # Chargement des poids du modèle GMF
    neumf_model.get_layer('mf_embedding_user').set_weights(model.get_layer('user_embedding').get_weights())
    neumf_model.get_layer('mf_embedding_item').set_weights(model.get_layer('item_embedding').get_weights())

    # Chargement des poids du modèle MLP
    neumf_model.get_layer('mlp_embedding_user').set_weights(mlp_model.get_layer('user_embedding').get_weights())
    neumf_model.get_layer('mlp_embedding_item').set_weights(mlp_model.get_layer('item_embedding').get_weights())

    for i in range(len(hidden_units)):
        neumf_model.get_layer(index=i + 3).set_weights(mlp_model.get_layer(index=i + 3).get_weights())

    # Compilation du modèle
    neumf_model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement du modèle
    neumf_model.fit([train_data['user_id'], train_data['movie_id']], train_data['rating'], epochs=epoch, batch_size=512,
                    validation_split=0.2)

    user_test = test_data['user_id']
    item_test = test_data['movie_id']
    # predictions = neumf_model.predict([user_test, item_test])
    # predicted_ratings = predictions * (max_rating - min_rating) + min_rating
    # print(predictions)
    # print(predicted_ratings)

    # Obtenez les prédictions du modèle NCF pour les notes des films
    predicted_rating = neumf_model.predict([user_test, item_test])
    predicted_ratings = predicted_rating * (max_rating - min_rating) + min_rating
    # Définissez un seuil pour la conversion des notes en étiquettes binaires
    threshold = 4

    # Convertissez les notes prédites en étiquettes binaires
    y_pred = [1 if rating >= threshold else 0 for rating in predicted_ratings]

    # Convertissez les vraies notes en étiquettes binaires
    y_true = [1 if rating >= threshold else 0 for rating in ratings_df['rating']]

    if len(y_true) > len(y_pred):
        y_true = y_true[:len(y_pred)]
    elif len(y_pred) > len(y_true):
        y_pred = y_pred[:len(y_true)]

    # print(len(y_true))
    # print(len(y_pred))
    print("mesures à epoch :", epoch, "et dim_latent : ", latent_dim)
    # Calculez le rappel
    recall = recall_score(y_true, y_pred, average='binary')

    print("Rappel : ", recall)

    # Calculer la précision
    precision = precision_score(y_true, y_pred)

    # Afficher le résultat
    print("Précision :", precision)

    # Calculer le NDCG
    ndcg = ndcg_score([y_true], [y_pred])

    # Afficher le résultat
    print("NDCG :", ndcg)

    # Calculer le NDCG
    ndcg_5 = ndcg_score([y_true], [y_pred], k=5)

    # Afficher le résultat
    print("NDCG@5 :", ndcg_5)
    recall_values.append(recall)
    precision_values.append(precision)
    ndcg_values.append(ndcg)

print("Rappel", recall_values)
print("Precision", precision_values)
print('NDCG', ndcg_values)

# Tracé du graphe
plt.plot(epochs, recall_values, label='Rappel')
plt.plot(epochs, precision_values, label='Précision', linewidth=2.8)
plt.plot(epochs, ndcg_values, label='NDCG', linewidth=1.2)

# Ajout de légendes et d'étiquettes
plt.xlabel('Nombre d\'itérations')
plt.ylabel('Métriques')
plt.title('Rappel, Précision et NDCG en fonction de nombre d\'epochs')
plt.legend()

# Affichage du graphe
plt.show()