import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

tmdb_df = pd.read_csv('data/tmdb_movies.csv')
tmdb_df = tmdb_df.drop(['original_language', 'budget', 'revenue', 
                        'runtime', 'status', 'tagline', 'vote_average', 
                        'vote_count', 'credits', 'backdrop_path', 
                        'recommendations'],
                        axis=1)
tmdb_df = tmdb_df.rename(columns={'id': 'tmdbId'})


# Read the CSV files into DataFrames
movies_df = pd.read_csv('data/ml-25m/movies.csv')
links_df = pd.read_csv('data/ml-25m/links.csv')

tmdb_df = tmdb_df.merge(links_df[['movieId', 'tmdbId']], on='tmdbId', how='left')
tmdb_df['movieId'] = tmdb_df['movieId'].fillna(0).astype(int)
tmdb_df = tmdb_df.dropna(subset=['genres'])

filtered_tmdb_df = tmdb_df[tmdb_df['movieId'] != 0]

sql_tmdb_df = filtered_tmdb_df.drop(['production_companies', 'keywords',],
                        axis=1)


# Create Movies DataFrame
sql_movies_df = sql_tmdb_df[['tmdbId', 'title', 'overview', 'popularity', 'release_date', 'poster_path']]


# Create Genres DataFrame
# First, we need to split the genres column into a list of genres
genres = sql_tmdb_df['genres'].str.split('-').explode().unique()
genres_df = pd.DataFrame(genres, columns=['genreName'])
genres_df['genreId'] = range(1, len(genres_df) + 1)


# Split the genres column into separate genres
sql_tmdb_df['genres'] = sql_tmdb_df['genres'].str.split('-')

# Use explode to create a new row for each genre for each movie
sql_tmdb_df = sql_tmdb_df.explode('genres')


# Create MovieGenres DataFrame
movie_genres_df = sql_tmdb_df[['tmdbId', 'genres']]
# Merge with genres_df to get genreId
movie_genres_df = movie_genres_df.merge(genres_df, left_on='genres',
                                        right_on='genreName')[[
                                            'tmdbId', 'genreId']]

sql_movies_df = sql_movies_df.drop_duplicates(subset='tmdbId', keep='first')
movie_genres_df = movie_genres_df.drop_duplicates(subset=['tmdbId', 'genreId'])

# Sort sql_movies_df by ascending tmdbId
remapped_sql_movies_df = sql_movies_df.sort_values('tmdbId')

# Create a new column 'movieId' with a continuous range
remapped_sql_movies_df['movieId'] = range(1, len(sql_movies_df) + 1)

# Merge movie_genres_df with remapped_sql_movies_df to get movieId
movie_genres_df = movie_genres_df.merge(remapped_sql_movies_df[['tmdbId', 'movieId']],
                                        on='tmdbId', how='left')

# Drop tmdbId column in movie_genres_df
movie_genres_df = movie_genres_df.drop('tmdbId', axis=1)

# Sort movie_genres_df by movieId and reorder columns
movie_genres_df = movie_genres_df.sort_values('movieId')
movie_genres_df = movie_genres_df[['movieId', 'genreId']]

# Reorder columns in remapped_sql_movies_df so that movieId is the first column
remapped_sql_movies_df = remapped_sql_movies_df[['movieId', 'tmdbId', 'title', 'overview',
                                                'popularity', 'release_date', 'poster_path']]

# Print that it is done
print("Movie and genre processing done")

ratings_df = pd.read_csv('data/ml-25m/ratings.csv')

ratings_df = ratings_df.merge(links_df[['movieId', 'tmdbId']], on='movieId',
                            how='left')
ratings_df['tmdbId'] = ratings_df['tmdbId'].fillna(0).astype(int)


filtered_ratings_df = ratings_df[ratings_df['tmdbId'] != 0]
filtered_ratings_df = filtered_ratings_df.drop(['movieId'], axis=1)

# Sort by 'userId', 'tmdbId', and 'timestamp' (descending) so the latest rating is first
filtered_ratings_df = filtered_ratings_df.sort_values(by=['userId', 'tmdbId', 'timestamp'], ascending=[True, True, False])

# Drop duplicates, keeping the first entry (which is now the latest rating due to sorting)
filtered_ratings_df = filtered_ratings_df.drop_duplicates(subset=['userId', 'tmdbId'], keep='first')

filtered_ratings_df['timestamp'] = pd.to_datetime(filtered_ratings_df['timestamp'], unit='s')

# Convert datetime to just date (without time)
filtered_ratings_df['timestamp'] = filtered_ratings_df['timestamp'].dt.date

filtered_ratings_df = filtered_ratings_df[filtered_ratings_df['tmdbId'].isin(sql_movies_df['tmdbId'])]

filtered_ratings_df = filtered_ratings_df.merge(remapped_sql_movies_df[['tmdbId', 'movieId']], on='tmdbId', how='left')

# Drop the tmdbId column
filtered_ratings_df = filtered_ratings_df.drop(['tmdbId'], axis=1)


# Sort the DataFrame by movieId
filtered_ratings_df = filtered_ratings_df.sort_values('movieId')

# Drop timestamp column
filtered_ratings_df = filtered_ratings_df.drop(['timestamp'], axis=1)

# Reorder the columns so that movieId is the first column
filtered_ratings_df = filtered_ratings_df[['userId', 'movieId', 'rating']]

print("Ratings processed")

tag_scores = pd.read_csv('data/ml-25m/genome-scores.csv')
tags_df = pd.read_csv('data/ml-25m/genome-tags.csv')

# Remove tag with id 742 from tag_scores DataFrame
tag_scores = tag_scores[tag_scores['tagId'] != 742]

# Remove tag with id 742 from tags_df DataFrame
tags_df = tags_df[tags_df['tagId'] != 742]

tags_df['tagId'] = range(1, len(tags_df) + 1)  # Start new index from 1

# Create a mapping from old tagId to new tagId
tag_id_mapping = dict(zip(tags_df['tagId'], tags_df['tagId']))

# Use this mapping to update tag_scores DataFrame
tag_scores['tagId'] = tag_scores['tagId'].map(tag_id_mapping)

# It's important to drop rows where tagId might have become NaN after mapping if any tagId did not have a mapping
tag_scores.dropna(subset=['tagId'], inplace=True)
tag_scores['tagId'] = tag_scores['tagId'].astype(int)

tag_scores = tag_scores.merge(links_df[['movieId', 'tmdbId']], on='movieId',
                            how='left')
tag_scores['tmdbId'] = tag_scores['tmdbId'].fillna(0).astype(int)

filtered_tag_scores = tag_scores[tag_scores['tmdbId'] != 0]
filtered_tag_scores = filtered_tag_scores.drop(['movieId'], axis=1)

filtered_tag_scores = filtered_tag_scores[['tmdbId', 'tagId', 'relevance']]

filtered_tag_scores = filtered_tag_scores.drop_duplicates(subset=['tmdbId', 'tagId'], keep='first')

filtered_tag_scores = filtered_tag_scores[filtered_tag_scores['tmdbId'].isin(sql_movies_df['tmdbId'])]

filtered_tag_scores = filtered_tag_scores.merge(remapped_sql_movies_df[['tmdbId', 'movieId']], on='tmdbId', how='left')

# Drop the tmdbId column
filtered_tag_scores = filtered_tag_scores.drop(['tmdbId'], axis=1)

# Reorder the columns so that movieId is the first column
filtered_tag_scores = filtered_tag_scores[['movieId', 'tagId', 'relevance']]

# Here we optimize the tag scores to keep only the top N tags per movie
filtered_tag_scores['rank'] = filtered_tag_scores.groupby('movieId')['relevance'].rank("dense", ascending=False)
filtered_tag_scores = filtered_tag_scores[filtered_tag_scores['rank'] <= 10].drop('rank', axis=1)

# Sort the DataFrame by movieId
filtered_tag_scores = filtered_tag_scores.sort_values('movieId')


# Print that tag scores have been processed
print("Tag scores processed")

num_ratings = filtered_ratings_df.groupby('movieId')['rating'].count()
num_movies_less_than_10_ratings = len(num_ratings[num_ratings < 10])
print(f"Number of movies with less than 10 ratings: {num_movies_less_than_10_ratings}")

num_movies_less_than_10_ratings = len(num_ratings[num_ratings < 5])
print(f"Number of movies with less than 5 ratings: {num_movies_less_than_10_ratings}")

# Filter out movies with less than 10 ratings
filtered_ratings_df = filtered_ratings_df[filtered_ratings_df['movieId'].isin(num_ratings[num_ratings >= 10].index)]

# Step 1: Take a random sample of 10% of filtered_ratings_df
sample_size = int(len(filtered_ratings_df) * 0.01)
sampled_ratings_df = filtered_ratings_df.sample(n=sample_size, random_state=42).copy()

# Step 2: Get the unique movie IDs from the sampled ratings
sampled_movie_ids = sampled_ratings_df['movieId'].unique()

# Step 3: Filter the other DataFrames based on the sampled movie IDs
sampled_movies_df = remapped_sql_movies_df[remapped_sql_movies_df['movieId'].isin(sampled_movie_ids)].copy()
sampled_movie_genres_df = movie_genres_df[movie_genres_df['movieId'].isin(sampled_movie_ids)].copy()
sampled_tag_scores_df = filtered_tag_scores[filtered_tag_scores['movieId'].isin(sampled_movie_ids)].copy()

# Step 4: Create mapping dictionaries for movie IDs and user IDs
old_to_new_movie_id = {old_id: new_id for new_id, old_id in enumerate(sorted(sampled_movie_ids), start=1)}
old_to_new_user_id = {old_id: new_id for new_id, old_id in enumerate(sorted(sampled_ratings_df['userId'].unique()), start=1)}

# Step 5: Remap the movie IDs and user IDs in the sampled ratings
sampled_ratings_df['old_movieId'] = sampled_ratings_df['movieId']
sampled_ratings_df['movieId'] = sampled_ratings_df['old_movieId'].map(old_to_new_movie_id)

sampled_ratings_df['old_userId'] = sampled_ratings_df['userId']
sampled_ratings_df['userId'] = sampled_ratings_df['old_userId'].map(old_to_new_user_id)

# Step 6: Apply the remapped movie IDs to the other cut-down DataFrames
sampled_movies_df['old_movieId'] = sampled_movies_df['movieId']
sampled_movies_df['movieId'] = sampled_movies_df['old_movieId'].map(old_to_new_movie_id)

sampled_movie_genres_df['old_movieId'] = sampled_movie_genres_df['movieId']
sampled_movie_genres_df['movieId'] = sampled_movie_genres_df['old_movieId'].map(old_to_new_movie_id)

sampled_tag_scores_df['old_movieId'] = sampled_tag_scores_df['movieId']
sampled_tag_scores_df['movieId'] = sampled_tag_scores_df['old_movieId'].map(old_to_new_movie_id)

sampled_movies_df = sampled_movies_df.drop(['old_movieId'], axis=1)
sampled_movie_genres_df = sampled_movie_genres_df.drop(['old_movieId'], axis=1)
sampled_ratings_df = sampled_ratings_df.drop(['old_movieId', 'old_userId'], axis=1)
sampled_tag_scores_df = sampled_tag_scores_df.drop(['old_movieId'], axis=1)

# Generate a single hash for a dummy password
dummy_password = "password123"
password_hash = generate_password_hash(dummy_password)

# Create unique dummy emails and passwords for each userId
users_df = pd.DataFrame(sampled_ratings_df['userId'].unique(), columns=['userId'])
users_df['email'] = users_df['userId'].apply(lambda x: f'user{x}@example.com')
# Assign the pre-generated hash to all users
users_df['password_hash'] = password_hash

print("Users processed")


# Merge the movies_in_ratings_df DataFrame with the genres_df DataFrame to get the genre names
movies_in_ratings_genres_df = sampled_movie_genres_df[sampled_movie_genres_df['movieId'].isin(sampled_ratings_df['movieId'])]

# Merge the movies_in_ratings_genres_df DataFrame with the genres_df DataFrame to get the genre names
movies_in_ratings_genres_df = movies_in_ratings_genres_df.merge(genres_df, on='genreId')

# Then, group by the 'genreName' column and count the number of unique movieIds in each group
genre_counts = movies_in_ratings_genres_df.groupby('genreName')['movieId'].nunique()

# Plot the result
plt.figure(figsize=(10, 6))
ax = genre_counts.plot(kind='bar')
plt.title('Number of Movies per Genre')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')

# Annotate the bars with their values
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()

def print_genre_counts(genre_counts):
    # Print the result
    print("Number of Movies per Genre:")
    for genre, count in genre_counts.items():
        print(f"Genre {genre}: {count}")

# Call the function
print_genre_counts(genre_counts)

# First, filter the tag scores DataFrame to only include movies present in the ratings DataFrame
movies_in_ratings_df = sampled_tag_scores_df[sampled_tag_scores_df['movieId'].isin(sampled_ratings_df['movieId'])]

# Then, group by the 'tagId' column and count the number of unique movieIds in each group
tag_counts = movies_in_ratings_df.groupby('tagId')['movieId'].nunique()

# Get the top 10 tags with the most counts
top_10_tag_counts = tag_counts.nlargest(10)

# Convert the top_10_tag_counts Series into a DataFrame
top_10_tag_counts_df = top_10_tag_counts.reset_index()

# Merge with the tags_df DataFrame to get the tag names
top_10_tag_counts_df = top_10_tag_counts_df.merge(tags_df, on='tagId')

# Set the 'tag' column as the index
top_10_tag_counts_df.set_index('tag', inplace=True)

# Plot the result
plt.figure(figsize=(10, 6))
ax = top_10_tag_counts_df['movieId'].plot(kind='bar')
plt.title('Number of Movies per Tag for Top 10 Tags')
plt.xlabel('Tag')
plt.ylabel('Number of Movies')

# Annotate the bars with their values
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()

def print_top_10_tag_counts(top_10_tag_counts_df):
    # Print the result
    print("Number of Movies per Tag for Top 10 Tags:")
    for tag, count in top_10_tag_counts_df['movieId'].items():
        print(f"Tag {tag}: {count}")

# Call the function
print_top_10_tag_counts(top_10_tag_counts_df)


# Get the bottom 10 tags with the least counts
bottom_10_tag_counts = tag_counts.nsmallest(10)

# Convert the bottom_10_tag_counts Series into a DataFrame
bottom_10_tag_counts_df = bottom_10_tag_counts.reset_index()

# Merge with the tags_df DataFrame to get the tag names
bottom_10_tag_counts_df = bottom_10_tag_counts_df.merge(tags_df, on='tagId')

# Set the 'tag' column as the index
bottom_10_tag_counts_df.set_index('tag', inplace=True)

# Plot the result
plt.figure(figsize=(10, 6))
ax = bottom_10_tag_counts_df['movieId'].plot(kind='bar')
plt.title('Number of Movies per Tag for Bottom 10 Tags')
plt.xlabel('Tag')
plt.ylabel('Number of Movies')

# Annotate the bars with their values
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()

# Count the occurrences of each rating
rating_counts = sampled_ratings_df['rating'].value_counts().sort_index()

# Plot the result
plt.figure(figsize=(10, 6))
ax = rating_counts.plot(kind='bar')
plt.title('Number of Ratings per Rating Value')
plt.xlabel('Rating')
plt.ylabel('Number of Ratings')

# Annotate the bars with their values
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))

plt.show()

def print_rating_counts(sampled_ratings_df):
    # Count the occurrences of each rating
    rating_counts = sampled_ratings_df['rating'].value_counts().sort_index()

    # Print the result
    print("Number of Ratings per Rating Value:")
    for rating, count in rating_counts.items():
        print(f"Rating {rating}: {count}")

# Call the function
print_rating_counts(sampled_ratings_df)

# Calculate average rating for each movie
average_ratings = filtered_ratings_df.groupby('movieId')['rating'].mean().round(2)

# Convert Series to DataFrame
average_ratings_df = average_ratings.reset_index()
num_ratings_df = num_ratings.reset_index()

# Rename the columns
average_ratings_df = average_ratings_df.rename(columns={'rating': 'average_rating'})
num_ratings_df = num_ratings_df.rename(columns={'rating': 'num_ratings'})

# Merge average_ratings_df and num_ratings_df with sampled_movies_df
sampled_movies_df = pd.merge(sampled_movies_df, average_ratings_df, on='movieId', how='left')
sampled_movies_df = pd.merge(sampled_movies_df, num_ratings_df, on='movieId', how='left')

# Boxplot of Ratings per Genre
ratings_genres_df = sampled_ratings_df.merge(sampled_movie_genres_df, on='movieId')
ratings_genres_df = ratings_genres_df.merge(genres_df, on='genreId')

# Create boxplot
sns.boxplot(x='genreName', y='rating', data=ratings_genres_df)
plt.title('Boxplot of Ratings per Genre')
plt.xlabel('Genre')
plt.ylabel('Rating')
plt.xticks(rotation=90)
plt.show()


# Merge tag scores with tags
tag_scores_tags_df = sampled_tag_scores_df.merge(tags_df, on='tagId')

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(tag_scores_tags_df['tag']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Get the maximum userId
max_userId = sampled_ratings_df['userId'].max()

# Get the number of unique userIds
num_unique_userIds = sampled_ratings_df['userId'].nunique()

# Assert that max_userId is not greater than num_unique_userIds
assert max_userId <= num_unique_userIds, "Max userId is greater than number of unique userIds"


max_movieId = sampled_ratings_df['movieId'].max()

num_unique_movieIds = sampled_ratings_df['movieId'].nunique()

assert max_movieId <= num_unique_movieIds, "Max movieId is greater than number of unique movieIds"


max_genreId = genres_df['genreId'].max()

num_unique_genreIds = genres_df['genreId'].nunique()

assert max_genreId <= num_unique_genreIds, "Max genreId is greater than number of unique genreIds"

max_tagId = tags_df['tagId'].max()

num_unique_tagIds = tags_df['tagId'].nunique()

assert max_tagId <= num_unique_tagIds, "Max tagId is greater than number of unique tagIds"


# Step 7: Save the DataFrames to CSV files
sampled_movies_df.to_csv('subset_data/sampled_movies.csv', index=False)
genres_df.to_csv('subset_data/genres.csv', index=False)
sampled_movie_genres_df.to_csv('subset_data/sampled_movie_genres.csv', index=False)
sampled_ratings_df.to_csv('subset_data/sampled_ratings.csv', index=False)
sampled_tag_scores_df.to_csv('subset_data/sampled_tag_scores.csv', index=False)
tags_df.to_csv('subset_data/tags.csv', index=False)
users_df.to_csv('subset_data/users.csv', index=False)