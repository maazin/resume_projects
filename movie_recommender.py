import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
import streamlit as st

# Step 1: Load the MovieLens dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Step 2: Merge movies and ratings datasets
data = pd.merge(ratings, movies, on='movieId')

# Step 3: Create a user-movie matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Step 4: Convert the user-movie matrix to a sparse matrix format
matrix = csr_matrix(user_movie_matrix.values)

# Step 5: Train a KNN model using collaborative filtering
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10)
model_knn.fit(matrix)

# Step 6: Define the recommendation function
def get_movie_recommendations(user_id, num_recommendations=5):
    user_index = user_id - 1  # Convert user ID to index
    distances, indices = model_knn.kneighbors(matrix[user_index], n_neighbors=6)  # Find similar users
    
    # Sum up ratings from similar users and get the top recommendations
    movie_indices = np.argsort(-user_movie_matrix.iloc[indices.flatten()].sum(axis=0))[:num_recommendations]
    recommended_movies = user_movie_matrix.columns[movie_indices].tolist()
    return recommended_movies

# Step 7: Streamlit interface for the recommendation system
st.title("Personalized Movie Recommendation System")
st.write("Enter a User ID to get movie recommendations based on similar users' preferences.")

# User input for User ID
user_id = st.number_input("Enter User ID:", min_value=1, max_value=data['userId'].nunique())

if st.button("Get Recommendations"):
    recommendations = get_movie_recommendations(user_id)
    st.write("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
