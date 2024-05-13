import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np

movies = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\Machine Learning\movies.csv')
ratings = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\Machine Learning\ratings.csv')

merged_data = pd.merge(ratings, movies, on='movieId')
merged_data = merged_data.drop(['timestamp', 'genres'], axis=1)
rating_matrix = merged_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

similarity_matrix = cosine_similarity(rating_matrix.values)
similarity_df = pd.DataFrame(similarity_matrix, index=rating_matrix.index, columns=rating_matrix.index)

def recommend_movies(user_id, num_recommendations, similarity_df, rating_matrix, movies):
    if user_id not in similarity_df.index:
        return "User ID not found."

    sim_scores = similarity_df[user_id]
    sorted_user_ids = sim_scores.sort_values(ascending=False).index
    
    
    sorted_user_ids = sorted_user_ids[sorted_user_ids != user_id]
    
    top_user_ratings = rating_matrix.loc[sorted_user_ids]

    
    weighted_ratings = np.dot(top_user_ratings.T, sim_scores[sorted_user_ids].values)
    sum_of_weights = np.abs(sim_scores[sorted_user_ids].values).sum()

    if sum_of_weights == 0:
        return "Insufficient data to make recommendations."

    weighted_average = weighted_ratings / sum_of_weights
    movie_recommendations = pd.Series(weighted_average, index=rating_matrix.columns)
    
    rated_movies = rating_matrix.loc[user_id] > 0
    movie_recommendations = movie_recommendations[~rated_movies]

    top_movies_ids = movie_recommendations.sort_values(ascending=False).head(num_recommendations).index
    top_movies = movies[movies['movieId'].isin(top_movies_ids)]

    return top_movies['title']


    
#User Interface
def main():
    

    while True:
        user_input = input("Enter a user ID to get recommendations (or 'quit' to stop): ")
        if user_input.lower() == 'quit':
            break
        try:
            user_id = int(user_input)
            recommendations = recommend_movies(user_id, 5, similarity_df, rating_matrix, movies)
            print("Recommended movies:")
            print(recommendations)
        except ValueError:
            print("Please enter a valid integer for the user ID.")

if __name__ == "__main__":
    main()









