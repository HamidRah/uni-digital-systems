# from enum import unique
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import pytorch_lightning as pl
# from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
# from torchmetrics import MeanSquaredError, MeanAbsoluteError
# import torch.nn.functional as F

# tag_scores_df = pd.read_csv('subset_data/sampled_tag_scores.csv')
# movie_genres_df = pd.read_csv('subset_data/sampled_movie_genres.csv')
# ratings_df = pd.read_csv('subset_data/sampled_ratings.csv')
# tags_df = pd.read_csv('subset_data/tags.csv')
# genres_df = pd.read_csv('subset_data/genres.csv')

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# # Create a PyTorch Dataset
# class MovieDataset(Dataset):
#     def __init__(self, ratings_df, tag_scores_df, movie_genres_df, max_tags=10):
#         self.ratings_df = ratings_df
#         self.tag_scores_df = tag_scores_df
#         self.max_tags = max_tags
#         self.movie_genres_df = movie_genres_df

#         self.precomputed_top_tags = self._precompute_top_tags()

#     def _precompute_top_tags(self):
#         # Structure to hold pre-computed top N tags and their relevance for each movie.
#         precomputed_top_tags = {}
#         for movie_id in self.ratings_df['movieId'].unique():
#             tags = self.tag_scores_df[self.tag_scores_df['movieId'] == movie_id]
#             tag_ids = tags['tagId'].tolist()
#             tag_rels = tags['relevance'].tolist()

#             # Ensure length matches max_tags, pad if necessary
#             tag_ids += [0] * (self.max_tags - len(tag_ids))
#             tag_rels += [0] * (self.max_tags - len(tag_rels))

#             precomputed_top_tags[movie_id] = (tag_ids[:self.max_tags], tag_rels[:self.max_tags])
        
#         return precomputed_top_tags

#     def __len__(self):
#         return len(self.ratings_df)

#     def __getitem__(self, index):
#         row = self.ratings_df.iloc[index]
#         user_id = torch.tensor(row['userId'], dtype=torch.long)
#         movie_id = torch.tensor(row['movieId'], dtype=torch.long)
#         rating = torch.tensor(row['rating'], dtype=torch.float)

#         # Retrieve the precomputed top N tags and relevance scores for the movie
#         tag_ids, tag_rel = self.precomputed_top_tags.get(row['movieId'], ([0]*self.max_tags, [0]*self.max_tags))

#         tag_ids = torch.tensor(tag_ids, dtype=torch.long)
#         tag_rel = torch.tensor(tag_rel, dtype=torch.float)


#         # Retrieve genre information
#         movie_genres = self.movie_genres_df[self.movie_genres_df['movieId'] == movie_id.item()]['genreId'].values
#         movie_genres = list(movie_genres[:10]) + [0] * (10 - len(movie_genres))  # Ensure fixed length for genres

#         movie_genres = torch.tensor(movie_genres, dtype=torch.long)
#         genre_mask = torch.tensor([1] * len(movie_genres) + [0] * (10 - len(movie_genres)), dtype=torch.float)

#         return user_id, movie_id, tag_ids, tag_rel, movie_genres, genre_mask, rating


# # Create a PyTorch Lightning Module
# class MovieRecommender(pl.LightningModule):
#     def __init__(self, num_users, num_movies, num_tags, num_genres, embedding_dim=64, max_tags=10):
#         super().__init__()
#         self.save_hyperparameters()

#         self.tag_scores_df = tag_scores_df
#         self.movie_genres_df = movie_genres_df
#         self.ratings_df = ratings_df
#         self.tags_df = tags_df
#         self.genres_df = genres_df

#         # Assuming embeddings are all of the same dimension
#         self.combined_embedding_dim = embedding_dim * 4  # Adjust according to your actual setup
#         self.fc_reduce = nn.Linear(self.combined_embedding_dim, embedding_dim)  # Reducing to match user embedding dimension

#         # Embedding layers
#         self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
#         self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
#         self.tag_embedding = nn.Embedding(num_tags + 1, embedding_dim)
#         self.genre_embedding = nn.Embedding(num_genres + 1, embedding_dim)

#         # Attention Layers
#         self.tag_attention = nn.Linear(embedding_dim, 1)
#         self.genre_attention = nn.Linear(embedding_dim, 1)

#         # Dropout Layer
#         self.dropout = nn.Dropout(0.5)

#         # Final Linear Layer
#         self.fc = nn.Linear(embedding_dim * 4, 1)


#         self.mse = MeanSquaredError()
#         self.mae = MeanAbsoluteError()

#         self.train_mse_scores = []
#         self.train_mae_scores = []
#         self.val_mse_scores = []
#         self.val_mae_scores = []



#     def forward(self, user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask):
#         # Embeddings
#         user_embedded = self.user_embedding(user_id)
#         movie_embedded = self.movie_embedding(movie_id)
#         tag_embedded = self.dropout(self.tag_embedding(tag_ids))
#         genre_embedded = self.dropout(self.genre_embedding(genre_indices))
        
#         # Apply attention to tags and genres
#         tag_weights = F.softmax(self.tag_attention(tag_embedded), dim=1)
#         genre_weights = F.softmax(self.genre_attention(genre_embedded), dim=1)
        
#         # Weight embeddings by their relevance scores and attention weights
#         tag_embedded = torch.sum(tag_embedded * tag_weights * tag_rel.unsqueeze(-1), dim=1)
#         genre_embedded = torch.sum(genre_embedded * genre_weights * genre_mask.unsqueeze(-1), dim=1) / torch.sum(genre_mask, dim=1, keepdim=True)
        
#         # Concatenate embeddings
#         concatenated = torch.cat((user_embedded, movie_embedded, tag_embedded, genre_embedded), dim=-1)
#         concatenated = self.dropout(concatenated)
        
#         # Final prediction
#         x = torch.relu(self.fc(concatenated))
#         return x.view(-1)


#     def training_step(self, batch, batch_idx):
#         user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, rating = batch
#         predicted_rating = self(user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask)

#         loss = nn.MSELoss()(predicted_rating, rating)

#         self.log('train_loss', loss)
#         self.log('train_mse', self.mse(predicted_rating, rating))
#         self.log('train_mae', self.mae(predicted_rating, rating))

#         self.train_mse_scores.append(self.mse(predicted_rating, rating))
#         self.train_mae_scores.append(self.mae(predicted_rating, rating))

#         return loss

#     def validation_step(self, batch, batch_idx):
#         user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, rating = batch
#         predicted_rating = self(user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask)

#         loss = nn.MSELoss()(predicted_rating, rating)

#         self.log('val_loss', loss)
#         self.log('val_mse', self.mse(predicted_rating, rating))
#         self.log('val_mae', self.mae(predicted_rating, rating))

#         self.val_mse_scores.append(self.mse(predicted_rating, rating))
#         self.val_mae_scores.append(self.mae(predicted_rating, rating))

#     def test_step(self, batch, batch_idx):
#         user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, rating = batch
#         predicted_rating = self(user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask)
        
#         loss = nn.MSELoss()(predicted_rating, rating)

#         self.log('test_loss', loss)
#         self.log('test_mse', self.mse(predicted_rating, rating))
#         self.log('test_mae', self.mae(predicted_rating, rating))

#     def on_train_epoch_end(self):
#         avg_train_mse = torch.stack(self.train_mse_scores).mean()
#         avg_train_mae = torch.stack(self.train_mae_scores).mean()
#         self.log('avg_train_mse', avg_train_mse, prog_bar=True)
#         self.log('avg_train_mae', avg_train_mae, prog_bar=True)
#         self.train_mse_scores.clear()
#         self.train_mae_scores.clear()
#         print(f'\n[Train] MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}')

#     def on_validation_epoch_end(self):
#         avg_val_mse = torch.stack(self.val_mse_scores).mean()
#         avg_val_mae = torch.stack(self.val_mae_scores).mean()
#         self.log('avg_val_mse', avg_val_mse, prog_bar=True)
#         self.log('avg_val_mae', avg_val_mae, prog_bar=True)
#         self.val_mse_scores.clear()
#         self.val_mae_scores.clear()
#         print(f'\n[Validation] MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}')


#     def configure_optimizers(self):
#         optimizer = optim.Adam(self.parameters(), lr=0.001)
#         lr_scheduler = {
#             'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98),
#             'name': 'learning_rate',
#             'interval': 'epoch',
#             'frequency': 1
#         }
#         return [optimizer], [lr_scheduler]
    

#     def recommend_top_n(self, user_id, n=10):
#         user_embedded = self.user_embedding(user_id).unsqueeze(0)  # Ensure the tensor is on the correct device

#         # Get unique movie IDs from your data
#         unique_movie_ids = self.ratings_df['movieId'].unique()

#         # Calculate embeddings for all movies
#         all_movie_embeddings = self.movie_embedding(torch.tensor(unique_movie_ids, dtype=torch.long).to(self.device))
#         all_tag_ids, all_tag_rels = zip(*(get_tags_for_movie(mid, self.tag_scores_df) for mid in unique_movie_ids))
#         all_genre_indices, all_genre_masks = zip(*(get_genres_for_movie(mid, self.movie_genres_df) for mid in unique_movie_ids))

#         # Convert lists to tensors and move to the same device as the model
#         all_tag_ids = torch.stack(all_tag_ids).to(self.device)
#         all_tag_rels = torch.stack(all_tag_rels).to(self.device)
#         all_genre_indices = torch.stack(all_genre_indices).to(self.device)
#         all_genre_masks = torch.stack(all_genre_masks).to(self.device)

#         # Compute tag and genre embeddings
#         all_tag_embeddings = self.tag_embedding(all_tag_ids)
#         all_genre_embeddings = self.genre_embedding(all_genre_indices)

#         # Apply attention to tags and genres
#         all_tag_weights = F.softmax(self.tag_attention(all_tag_embeddings), dim=2)
#         all_genre_weights = F.softmax(self.genre_attention(all_genre_embeddings), dim=2)

#         # Aggregate embeddings with attention and relevance/mask
#         all_tag_aggregated = torch.sum(all_tag_embeddings * all_tag_weights * all_tag_rels.unsqueeze(-1), dim=1)
#         all_genre_aggregated = torch.sum(all_genre_embeddings * all_genre_weights * all_genre_masks.unsqueeze(-1), dim=1) / torch.sum(all_genre_masks, dim=1, keepdim=True)

#         # Calculate embeddings for all movies and concatenate
#         all_composite_embeddings = torch.cat([all_movie_embeddings, all_tag_aggregated, all_genre_aggregated], dim=1)

#         # Adjust the dimensionality of all_composite_embeddings to match the input dimensionality of fc_reduce
#         all_composite_embeddings = all_composite_embeddings.view(-1, self.fc_reduce.in_features)

#         # Reduce dimensionality to match user embeddings if needed
#         all_composite_embeddings = self.fc_reduce(all_composite_embeddings)

#         # Compute similarities
#         similarities = torch.matmul(user_embedded, all_composite_embeddings.t()).squeeze(0)  # Remove batch dimension
#         _, top_movie_indices = torch.topk(similarities, n)

#         return top_movie_indices
    
#     def compute_tag_attention_weights(self, tag_ids, tag_embedded):
#         tag_weights = F.softmax(self.tag_attention(tag_embedded), dim=1)
#         return tag_weights

#     def compute_genre_attention_weights(self, genre_embedded):
#         genre_weights = F.softmax(self.genre_attention(genre_embedded), dim=1)
#         return genre_weights

#     def predict_ratings(self, user_id, movie_ids):
#         user_id = user_id.unsqueeze(0).repeat(len(movie_ids), 1).squeeze(1)
        
#         # Retrieve tags and genres for each movie
#         tag_ids, tag_rels = zip(*(get_tags_for_movie(mid.item(), self.tag_scores_df) for mid in movie_ids))
#         genre_indices, genre_masks = zip(*(get_genres_for_movie(mid.item(), self.movie_genres_df) for mid in movie_ids))

#         # Convert to tensors and transfer to the appropriate device
#         tag_ids = torch.stack(tag_ids).to(self.device)
#         tag_rels = torch.stack(tag_rels).to(self.device)
#         genre_indices = torch.stack(genre_indices).to(self.device)
#         genre_masks = torch.stack(genre_masks).to(self.device)

#         # Calculate predicted ratings using the model's forward pass
#         predicted_ratings = self.forward(user_id, movie_ids, tag_ids, tag_rels, genre_indices, genre_masks)

#         # Embed tags and genres, then compute attention weights
#         tag_embedded = self.tag_embedding(tag_ids)
#         genre_embedded = self.genre_embedding(genre_indices)
#         tag_attn_weights = self.compute_tag_attention_weights(tag_ids, tag_embedded)
#         genre_attn_weights = self.compute_genre_attention_weights(genre_embedded)

#         # Get top tag and genre indices
#         top_tag_indices = torch.topk(tag_attn_weights, k=3, dim=1).indices + 1
#         top_genre_indices = torch.topk(genre_attn_weights, k=3, dim=1).indices + 1

#         # Retrieve tag and genre names
#         tag_names = [self.tags_df[self.tags_df['tagId'] == idx.item()]['tag'].values[0] for idx in top_tag_indices.flatten()]
#         genre_names = [self.genres_df[self.genres_df['genreId'] == idx.item()]['genreName'].values[0] for idx in top_genre_indices.flatten()]

#         results = []
#         # Prepare a structured output for each movie
#         for i in range(len(movie_ids)):
#             movie_info = {
#                 "movie_id": movie_ids[i].item(),
#                 "predicted_rating": float(predicted_ratings[i].item()),
#                 "tags": tag_names[i*3:(i+1)*3],
#                 "genres": genre_names[i*3:(i+1)*3]
#             }
#             results.append(movie_info)
        
#         return results
    


# def get_tags_for_movie(movie_id, tag_scores_df, max_tags=10):
#     movie_tags = tag_scores_df[tag_scores_df['movieId'] == movie_id].nlargest(max_tags, 'relevance')
#     tag_ids = movie_tags['tagId'].tolist()
#     tag_rel = movie_tags['relevance'].tolist()
#     # Pad if necessary
#     tag_ids += [0] * (max_tags - len(tag_ids))
#     tag_rel += [0.0] * (max_tags - len(tag_rel))
#     return torch.tensor(tag_ids, dtype=torch.long), torch.tensor(tag_rel, dtype=torch.float)


# def get_genres_for_movie(movie_id, movie_genres_df, max_genres=10):
#     movie_genres = movie_genres_df[movie_genres_df['movieId'] == movie_id]['genreId'].values
#     movie_genres = list(movie_genres[:max_genres])  # Truncate if necessary
#     movie_genres = movie_genres + [0] * (max_genres - len(movie_genres))  # Pad with zeros
#     genre_mask = [1.0 if i < len(movie_genres) else 0.0 for i in range(max_genres)]  # Mask to distinguish real genres from padding
#     return torch.tensor(movie_genres, dtype=torch.long), torch.tensor(genre_mask, dtype=torch.float)


# def load_data(max_tags=10):
#     # Load the data
#     movies_df = pd.read_csv('subset_data/sampled_movies.csv')
#     genres_df = pd.read_csv('subset_data/genres.csv')
#     movie_genres_df = pd.read_csv('subset_data/sampled_movie_genres.csv')
#     ratings_df = pd.read_csv('subset_data/sampled_ratings.csv')
#     tag_scores = pd.read_csv('subset_data/sampled_tag_scores.csv')
#     tags_df = pd.read_csv('subset_data/tags.csv')

#     # Merge tag_scores with tags_df to bring in the 'tag' column
#     tag_scores = tag_scores.merge(tags_df, on='tagId', how='left')


#     return movies_df, genres_df, movie_genres_df, ratings_df, tag_scores, tags_df

# # ********************************************************************************************************************

# def main():

#     movies_df, genres_df, movie_genres_df, ratings_df, tag_scores, tags_df = load_data()

#     # Create a mapping for movieIds
#     unique_movies = ratings_df['movieId'].unique()

#     # Similarly, for userId
#     unique_users = ratings_df['userId'].unique()

#     num_users = len(unique_users)
#     num_movies = len(unique_movies)
#     num_genres = len(genres_df)
#     num_tags = len(tags_df)

#     print(f"Total number of ratings: {len(ratings_df)}")
#     print(f"Number of users: {num_users}, Number of movies: {num_movies}, Number of tags: {num_tags}")
#     print(f"Number of genres: {num_genres}")

#     torch.set_float32_matmul_precision('medium')

#     train_df = ratings_df.sample(frac=0.8, random_state=42)
#     val_df = ratings_df.drop(train_df.index)

#     train_dataset = MovieDataset(train_df, tag_scores, movie_genres_df)
#     val_dataset = MovieDataset(val_df, tag_scores, movie_genres_df)

#     train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=8, pin_memory=True, persistent_workers=True)

#     model = MovieRecommender(num_users=num_users, num_movies=num_movies, num_tags=num_tags, num_genres=num_genres, embedding_dim=64).to('cuda')

#     assert model.user_embedding.num_embeddings >= num_users, f"User ID out of range. Maximum user ID: {model.user_embedding.num_embeddings}"
#     assert model.movie_embedding.num_embeddings >= num_movies, f"Movie ID out of range. Maximum movie ID: {model.movie_embedding.num_embeddings}"
#     assert model.tag_embedding.num_embeddings >= num_tags, f"Tag ID out of range. Maximum tag ID: {model.tag_embedding.num_embeddings}"
#     assert model.genre_embedding.num_embeddings >= num_genres, f"Genre ID out of range. Maximum genre ID: {model.genre_embedding.num_embeddings}"

#     trainer = pl.Trainer(
#     max_epochs=20,
#     devices=1,
#     accelerator='gpu',
#     precision='16-mixed',
#     callbacks=[
#         EarlyStopping(monitor='val_loss', patience=3, mode='min'),
#         ModelCheckpoint(dirpath='checkpoints', filename='best-checkpoint', monitor='val_loss', mode='min', save_top_k=1),
#         LearningRateMonitor(logging_interval='epoch')
#     ]
# )
    
#     trainer.fit(model, train_dataloader, val_dataloader)

#     torch.save(model.state_dict(), './model_store/model.pt')


# if __name__ == '__main__':
#     main()