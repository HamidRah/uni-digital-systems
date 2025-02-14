from sqlalchemy import create_engine, text
import pandas as pd

# Create a connection to your MySQL database
# Replace 'username', 'password', 'localhost', 'database_name' with your actual values
engine = create_engine(
    'mysql+mysqlconnector://hamid:U9kknW3Q^8@localhost:3307/moviemanager_db')

# Read CSV files and create DataFrames
remapped_sql_movies_df = pd.read_csv('../app/subset_data/sampled_movies.csv')
genres_df = pd.read_csv('../app/subset_data/genres.csv')
movie_genres_df = pd.read_csv('../app/subset_data/sampled_movie_genres.csv')
users_df = pd.read_csv('../app/subset_data/users.csv')
tags_df = pd.read_csv('../app/subset_data/tags.csv')
filtered_tag_scores = pd.read_csv('../app/subset_data/sampled_tag_scores.csv')
filtered_ratings_df = pd.read_csv('../app/subset_data/sampled_ratings.csv')


    # Continue with the rest of the code...

# Write DataFrames to MySQL tables
remapped_sql_movies_df.to_sql('movies', con=engine, if_exists='append', index=False)
print("Movies inserted")
genres_df.to_sql('genres', con=engine, if_exists='append', index=False)
print("Genres inserted")
movie_genres_df.to_sql('movie_genres', con=engine, if_exists='append', index=False)
print("Movie genres inserted")
users_df.to_sql('users', con=engine, if_exists='append', index=False)
print("Users inserted")
tags_df.to_sql('tags', con=engine, if_exists='append', index=False)
print("Tags inserted")

def insert_in_chunks_ratings(dataframe, chunk_size=10000):
    num_chunks = len(dataframe) // chunk_size + (1 if len(dataframe) % chunk_size else 0)
    for i in range(num_chunks):
        chunk = dataframe[i*chunk_size:(i+1)*chunk_size]
        chunk.to_sql(name='ratings', con=engine, if_exists='append', index=False)
        print(f"Inserted chunk {i+1}/{num_chunks}")
    print("All chunks inserted")


def insert_in_chunks_tags_scores(dataframe, chunk_size=10000):
    num_chunks = len(dataframe) // chunk_size + (1 if len(dataframe) % chunk_size else 0)
    for i in range(num_chunks):
        chunk = dataframe[i*chunk_size:(i+1)*chunk_size]
        chunk.to_sql(name='tag_scores', con=engine, if_exists='append', index=False)
        print(f"Inserted chunk {i+1}/{num_chunks}")
    print("All chunks inserted")

insert_in_chunks_tags_scores(filtered_tag_scores, chunk_size=10000)
insert_in_chunks_ratings(filtered_ratings_df, chunk_size=10000)

# def insert_in_chunks_ratings_optimized(dataframe, chunk_size=1000):
#     with engine.begin() as conn:  # automatically commits or rolls back
#         # Temporarily disable foreign key checks
#         conn.execute(text("SET foreign_key_checks = 0;"))
#         num_chunks = len(dataframe) // chunk_size + (1 if len(dataframe) % chunk_size else 0)
#         for i in range(num_chunks):
#             chunk = dataframe.iloc[i*chunk_size:(i+1)*chunk_size]
#             chunk.to_sql(name='ratings', con=conn, if_exists='append', index=False, method='multi')
#             print(f"Inserted chunk {i+1}/{num_chunks}")
#         # Re-enable foreign key checks
#         conn.execute(text("SET foreign_key_checks = 1;"))
#     print("All chunks inserted")

# Call the function to insert data
# insert_in_chunks_ratings_optimized(filtered_ratings_df, chunk_size=10000)