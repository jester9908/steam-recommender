import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

#set display options to show all columns
pd.set_option('display.max_columns', None) 

# Load datasets
reviews = pd.read_csv('steam_reviews.csv')

# summary statistics of the dataset
print(reviews.describe())

#Plot histograms for numerical features
numerical_features = ['votes_helpful', 'weighted_vote_score', 'author.playtime_forever', 'author.playtime_last_two_weeks']
reviews[numerical_features].hist(bins=15, figsize=(10, 8))
plt.tight_layout()
plt.show()

# Plot distribution of weighted_vote_score
sns.histplot(reviews['weighted_vote_score'], kde=True)
plt.title('Distribution of Weighted Vote Score')
plt.show()

# Check for missing values and duplicates and data cleaning
reviews.dropna(subset=['review_id'], inplace=True) # Drop rows with missing review_id
reviews.dropna(subset=['review'], inplace=True) # Drop rows with missing review
reviews.drop_duplicates(subset='review_id', inplace=True) # Drop duplicates

# Fill missing data with 0
reviews['author.playtime_at_review'].fillna(0, inplace=True) 
reviews['author.playtime_forever'].fillna(0, inplace=True) 
reviews['author.last_played'].fillna(0, inplace=True)
reviews['author.playtime_last_two_weeks'].fillna(0, inplace=True)

# print(reviews.isnull().sum())

# Convert columns to datetime
reviews['timestamp_created'] = pd.to_datetime(reviews['timestamp_created'], unit='s')
reviews['timestamp_updated'] = pd.to_datetime(reviews['timestamp_updated'], unit='s')
reviews['author.last_played'] = pd.to_datetime(reviews['author.last_played'], unit='s')

# print(reviews.info())

# Filter only English reviews
reviews_eng = reviews[reviews['language'] == 'english']

# Clean text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

reviews_eng['review'] = reviews_eng['review'].apply(clean_text)

# handles inconsistent boolean values
reviews_eng = reviews_eng[reviews_eng['recommended'].isin([True, False])]
reviews_eng = reviews_eng[reviews_eng['steam_purchase'].isin([True, False])]
reviews_eng = reviews_eng[reviews_eng['received_for_free'].isin([True, False])]
reviews_eng = reviews_eng[reviews_eng['written_during_early_access'].isin([True, False])]

# Filter data for modeling based on games the author has played
df_played = reviews_eng[reviews_eng['author.playtime_forever'] > 0]

# print(df_played.head)

#create a list of unique app_ids with their aggregated weighted_vote_score and review count.
app_ranking = reviews_eng.groupby('app_id').agg({'weighted_vote_score': 'mean', 'review_id': 'count'}).reset_index()
app_ranking.columns = ['app_id','mean_weighted_vote_score', 'total_review_count']

unique_app_names = reviews_eng[['app_id', 'app_name']].drop_duplicates()
app_ranking = pd.merge(app_ranking, unique_app_names, on='app_id', how='left')
app_ranking = app_ranking[['app_id', 'app_name', 'mean_weighted_vote_score', 'total_review_count']]
app_ranking = app_ranking.sort_values('mean_weighted_vote_score', ascending=False)

# print(app_ranking.head(10))

# Normalize numerical features
scaler = MinMaxScaler()
app_ranking[['mean_weighted_vote_score', 'total_review_count']] = scaler.fit_transform(app_ranking[['mean_weighted_vote_score', 'total_review_count']])

# Create a user-item matrix
user_item_matrix = df_played.pivot_table(index='author.steamid', columns='app_id', values='weighted_vote_score').fillna(0)

# Convert the matrix to a sparse format for efficiency
user_item_sparse = csr_matrix(user_item_matrix.values)

#Collaberative filtering
# Apply SVD to decompose the user-item matrix
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_item_sparse)  # User latent factors
item_factors = svd.components_.T  # Item latent factors

def recommend_games(user_id, num_recommendations=10):
    user_index = user_item_matrix.index.get_loc(user_id) # Get the index of the user in the user_item_matrix
    scores = np.dot(user_factors[user_index, :], item_factors.T) # Calculate the predicted scores for all games
    top_indices = np.argsort(scores)[::-1][:num_recommendations] # Get indices of the top N scores
    top_app_ids = user_item_matrix.columns[top_indices] # Map indices to app_ids
    recommendations = app_ranking[app_ranking['app_id'].isin(top_app_ids)] # Return the recommended games with app_id, app_name, and other relevant information
    return recommendations

# Example: Get top 10 recommendations for a specific user
recommended_games = recommend_games(76561198995830324)
print(recommended_games)


'''
Example user ids
76561198074088375
76561198995830324
'''