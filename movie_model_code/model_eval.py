import pandas as pd
import torch

from sklearn.metrics import mean_absolute_error, r2_score

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from model_weight_train import MovieRecommender

import matplotlib.pyplot as plt

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

tag_scores_df = pd.read_csv('app/subset_data/sampled_tag_scores.csv')
movie_genres_df = pd.read_csv('app/subset_data/sampled_movie_genres.csv')
ratings_df = pd.read_csv('app/subset_data/sampled_ratings.csv')
tags_df = pd.read_csv('app/subset_data/tags.csv')
genres_df = pd.read_csv('app/subset_data/genres.csv')
    
# Load the model from the checkpoint
model = MovieRecommender.load_from_checkpoint(checkpoint_path='checkpoints/best-checkpoint.ckpt')
model = model.to('cpu')
# Print the model's state_dict (parameters)
# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# # Print the model's hyperparameters
# print("Model's hyperparameters:")
# print(model.hparams)


# Filter users who have more than 5 ratings
user_counts = ratings_df['userId'].value_counts()
user_ids = user_counts[user_counts > 5].index

# Calculate average ratings for these users
average_ratings = ratings_df[ratings_df['userId'].isin(user_ids)].groupby('userId')['rating'].mean()

# Find the user with the most ratings
user_id = ratings_df['userId'].value_counts().idxmax()

# Find the user with the lowest average rating
user_id = average_ratings.idxmin()

# Highest average rating
user_id = average_ratings.idxmax()

# Select the rows for this user
subset_df = ratings_df[ratings_df['userId'] == user_id]

# Get the movie IDs
movie_ids = subset_df['movieId'].values

# Convert to tensors
# Convert to tensors
user_id_tensor = torch.tensor([user_id], dtype=torch.long).long()
movie_ids_tensor = torch.tensor(movie_ids, dtype=torch.long).long()

# Make predictions
model.eval()
with torch.no_grad():
    predictions = model.predict_ratings(movie_ids_tensor, user_id_tensor)

# Extract predicted ratings from the results
predicted_ratings = [result['predicted_rating'] for result in predictions]

# Convert predicted ratings to a Tensor
predictions_tensor = torch.tensor(predicted_ratings)

# Get the actual ratings
actual_ratings = torch.tensor(subset_df['rating'].values)

# Compute the Mean Squared Error
mse = ((predictions_tensor - actual_ratings) ** 2).mean().item()
print(f'MSE: {mse}')

# Convert mse to a Tensor
mse_tensor = torch.tensor(mse)

# Compute the Root Mean Squared Error
rmse = torch.sqrt(mse_tensor)
print(f'RMSE: {rmse.item()}')

# Compute the Mean Absolute Error
mae = mean_absolute_error(actual_ratings, predictions_tensor)
print(f'MAE: {mae}')

# Compute the R-squared score
r2 = r2_score(actual_ratings, predictions_tensor)
print(f'R²: {r2}')

for i in range(len(movie_ids)):  # Change this value to print more or fewer comparisons
    print(f'Predicted rating: {predictions_tensor[i]}, Actual rating: {actual_ratings[i]}')


# Create lists for the performance metrics
user_categories = ['Most Ratings', 'Lowest Avg Ratings', 'Highest Avg Ratings']
weighted_mse = [0.401420781463534, 0.5409529726997264, 1.2196430081075333]
weighted_rmse = [0.6335777640342712, 0.7354950308799744, 1.1043744087219238]
weighted_mae = [0.47215739641775634, 0.5791757330298424, 0.9539652585983276]
weighted_r2 = [0.2482803472312628, -1.8850825210652076, 0.0]

non_weighted_mse = [0.37422747485291885, 2.505586094358401, 0.2518775512341563]
non_weighted_rmse = [0.6117413640022278, 1.58290433883667, 0.5018740296363831]
non_weighted_mae = [0.4703771575203155, 1.4684060662984848, 0.4412569999694824]
non_weighted_r2 = [0.2992038268987508, -12.36312583657814, 0.0]

baseline_rmse = [0.5754, 1.2873, 0.6585]
baseline_mae = [0.4386, 1.1873, 0.5685]
baseline_r2 = [0.3650647455576089, -9.607689691801326, 0.0]

# Set the width of the bars
bar_width = 0.25

# Set the positions of the bars on the x-axis
r1 = range(len(user_categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create the bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

bars1 = ax1.bar(r1, weighted_rmse, width=bar_width, label='Weighted RMSE')
bars2 = ax1.bar(r2, non_weighted_rmse, width=bar_width, label='Non-Weighted RMSE')
bars3 = ax1.bar(r3, baseline_rmse, width=bar_width, label='Baseline RMSE')

ax1.set_xlabel('User Categories')
ax1.set_ylabel('RMSE')
ax1.set_title('Comparison of RMSE')
ax1.set_xticks([r + bar_width for r in range(len(user_categories))])
ax1.set_xticklabels(user_categories)
ax1.legend()

bars4 = ax2.bar(r1, weighted_mae, width=bar_width, label='Weighted MAE')
bars5 = ax2.bar(r2, non_weighted_mae, width=bar_width, label='Non-Weighted MAE')
bars6 = ax2.bar(r3, baseline_mae, width=bar_width, label='Baseline MAE')

ax2.set_xlabel('User Categories')
ax2.set_ylabel('MAE')
ax2.set_title('Comparison of MAE')
ax2.set_xticks([r + bar_width for r in range(len(user_categories))])
ax2.set_xticklabels(user_categories)
ax2.legend()

# Add values on top of each bar
for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
    for bar in bars:
        yval = bar.get_height()
        ax = bar.axes
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# r2 bar chart
r1 = range(len(user_categories))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create the bar chart
fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(r1, weighted_r2, width=bar_width, label='Weighted R²')
bars2 = ax.bar(r2, non_weighted_r2, width=bar_width, label='Non-Weighted R²')
bars3 = ax.bar(r3, baseline_r2, width=bar_width, label='Baseline R²')

ax.set_xlabel('User Categories')
ax.set_ylabel('R²')
ax.set_title('Comparison of R²')
ax.set_xticks([r + bar_width for r in range(len(user_categories))])
ax.set_xticklabels(user_categories)
ax.legend()

# Add values on top of each bar
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        yval = bar.get_height()
        ax = bar.axes
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

plt.tight_layout()
plt.show()



# Prepare your data
reader = Reader(rating_scale=(ratings_df['rating'].min(), ratings_df['rating'].max()))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)

# Split the dataset into train and test
trainset, testset = train_test_split(data, test_size=.25)

# Train the SVD model
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

# Compute MAE
accuracy.mae(predictions)



# Filter the test set to only include ratings from the user with the most ratings
testset_filtered = ratings_df[ratings_df['userId'] == user_id]

# Convert the DataFrame to a list of tuples
testset_filtered = [tuple(x) for x in testset_filtered.to_records(index=False)]

# Make predictions for this user
predictions = model.test(testset_filtered)

# Then compute RMSE
print(accuracy.rmse(predictions))

# Compute MAE
print(accuracy.mae(predictions))

actual_ratings = [pred.r_ui for pred in predictions]
predicted_ratings = [pred.est for pred in predictions]

# Compute the R² score
r2 = r2_score(actual_ratings, predicted_ratings)
print(f'Baseline R²: {r2}')
