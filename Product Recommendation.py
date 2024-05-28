import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

df=pd.read_csv('ratings_Beauty.csv')
print(df.head(10))

df=df.drop('Timestamp',axis=1)
LE=LabelEncoder()
df['UserId']=LE.fit_transform(df['UserId'])
df['ProductId']=LE.fit_transform(df['ProductId'])
print(df)

train,test=train_test_split(df,test_size=0.2)
print(train)

no_of_rated_products_per_user = df.groupby(by='UserId')['Rating'].count().sort_values(ascending=False)
print(no_of_rated_products_per_user.head(25))

nearest_neighbors=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=5)
print(nearest_neighbors)

nearest_neighbors.fit(train[['UserId','ProductId']])
print(nearest_neighbors)

#Recommendation
def recommendation_system_function(userID, n_recommendation, nearest_neighbors):
    user_profile = df[df['UserId'] == userID].drop('Rating', axis=1)

    
    distances, indices = nearest_neighbors.kneighbors(user_profile)
    similar_users = indices.flatten()

   
    user_recommendations = []
    for user in similar_users:
        products_rated_by_user = df[df['UserId'] == user]
        product_list = products_rated_by_user['ProductId'].unique()
        user_recommendations.extend(product_list)
    unique_recommendations = list(set(user_recommendations))

   
    product_ratings_count = df[df['ProductId'].isin(unique_recommendations)]['ProductId'].value_counts()
    sorted_recommendations = product_ratings_count.sort_values(ascending=False)
    top_recommendations = sorted_recommendations.index[:n_recommendation]
    recommended_product_codes = LE.inverse_transform(top_recommendations)

    recommendation_table = pd.DataFrame({'Product ID': top_recommendations, 'Product Code': recommended_product_codes})
    recommendation_table['Rating Count'] = recommendation_table['Product ID'].map(product_ratings_count)
    recommendation_table.set_index('Product ID', inplace=True)

    print("Recommended Products:")
    print(recommendation_table)

    return recommended_product_codes
    print(df[df['ProductId']=='B003BMGA6C']['UserId'])

User_Id = 56  
num_recommendations = 5 

recommended_products = recommendation_system_function(User_Id, num_recommendations, nearest_neighbors)
print(recommended_products)

