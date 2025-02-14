import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, MeanAbsoluteError
import torch.nn.functional as F

tag_scores_df = pd.read_csv('app/subset_data/sampled_tag_scores.csv')
movie_genres_df = pd.read_csv('app/subset_data/sampled_movie_genres.csv')
ratings_df = pd.read_csv('app/subset_data/sampled_ratings.csv')
tags_df = pd.read_csv('app/subset_data/tags.csv')
genres_df = pd.read_csv('app/subset_data/genres.csv')


# Calculate weights based on the inverse of rating frequencies
rating_counts = {0.5: 3768, 1.0: 7717, 1.5: 3909, 2.0: 16194, 2.5: 12563, 3.0: 48348, 3.5: 31313, 4.0: 66219, 4.5: 22103, 5.0: 35665}
total_ratings = sum(rating_counts.values())
weights = {rating: total_ratings / count for rating, count in rating_counts.items()}


weight_tensor = torch.tensor(list(weights.values()), dtype=torch.float)

# Create a PyTorch Dataset
class MovieDataset(Dataset):
    def __init__(self, ratings_df, tag_scores_df, movie_genres_df, weights, max_tags=10):
        self.ratings_df = ratings_df
        self.tag_scores_df = tag_scores_df
        self.max_tags = max_tags
        self.movie_genres_df = movie_genres_df
        self.weights = weights

        self.precomputed_top_tags = self._precompute_top_tags()

    def _precompute_top_tags(self):
        # Structure to hold pre-computed top N tags and their relevance for each movie.
        precomputed_top_tags = {}
        for movie_id in self.ratings_df['movieId'].unique():
            tags = self.tag_scores_df[self.tag_scores_df['movieId'] == movie_id]
            tag_ids = tags['tagId'].tolist()
            tag_rels = tags['relevance'].tolist()

            # Ensure length matches max_tags, pad if necessary
            tag_ids += [0] * (self.max_tags - len(tag_ids))
            tag_rels += [0] * (self.max_tags - len(tag_rels))

            precomputed_top_tags[movie_id] = (tag_ids[:self.max_tags], tag_rels[:self.max_tags])
        
        return precomputed_top_tags

    def __len__(self):
        return len(self.ratings_df)

    def __getitem__(self, index):
        row = self.ratings_df.iloc[index]
        user_id = torch.tensor(row['userId'], dtype=torch.long)
        movie_id = torch.tensor(row['movieId'], dtype=torch.long)
        rating = torch.tensor(row['rating'], dtype=torch.float)

        # Retrieve the precomputed top N tags and relevance scores for the movie
        tag_ids, tag_rel = self.precomputed_top_tags.get(row['movieId'], ([0]*self.max_tags, [0]*self.max_tags))

        tag_ids = torch.tensor(tag_ids, dtype=torch.long)
        tag_rel = torch.tensor(tag_rel, dtype=torch.float)


        # Retrieve genre information
        movie_genres = self.movie_genres_df[self.movie_genres_df['movieId'] == movie_id.item()]['genreId'].values
        movie_genres = list(movie_genres[:10]) + [0] * (10 - len(movie_genres))  # Ensure fixed length for genres

        movie_genres = torch.tensor(movie_genres, dtype=torch.long)
        genre_mask = torch.tensor([1] * len(movie_genres) + [0] * (10 - len(movie_genres)), dtype=torch.float)

        weight = torch.tensor(weights[row['rating']], dtype=torch.float)

        return user_id, movie_id, tag_ids, tag_rel, movie_genres, genre_mask, rating, weight


# Create a PyTorch Lightning Module
class MovieRecommender(pl.LightningModule):
    def __init__(self, num_users, num_movies, num_tags, num_genres, embedding_dim, weight_tensor, optimal_params):
        super().__init__()
        self.save_hyperparameters()

        self.tag_scores_df = tag_scores_df
        self.movie_genres_df = movie_genres_df
        self.ratings_df = ratings_df
        self.tags_df = tags_df
        self.genres_df = genres_df

        self.weight_tensor = weight_tensor

        self.optimal_params = optimal_params

        self.combined_embedding_dim = embedding_dim * 4  
        self.fc_reduce = nn.Linear(self.combined_embedding_dim, embedding_dim)  # Reducing to match user embedding dimension

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)
        self.tag_embedding = nn.Embedding(num_tags + 1, embedding_dim)
        self.genre_embedding = nn.Embedding(num_genres + 1, embedding_dim)

        # Attention Layers
        self.tag_attention = nn.Linear(embedding_dim, 1)
        self.genre_attention = nn.Linear(embedding_dim, 1)

        # Dropout Layer
        self.dropout = nn.Dropout(0.5)

        # Final Linear Layer
        self.fc = nn.Linear(embedding_dim * 4, 1)


        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()

        self.train_mse_scores = []
        self.train_mae_scores = []
        self.val_mse_scores = []
        self.val_mae_scores = []



    def forward(self, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, user_id=None):
        # Embeddings
        movie_embedded = self.movie_embedding(movie_id)
        tag_embedded = self.dropout(self.tag_embedding(tag_ids))
        genre_embedded = self.dropout(self.genre_embedding(genre_indices))
        
        # If a user_id is provided, use it to generate user embeddings
        if user_id is not None:
            user_embedded = self.user_embedding(user_id)
        else:
            # If no user_id is provided, use the movie embeddings in place of the user embeddings
            user_embedded = movie_embedded

        # Apply attention to tags and genres
        tag_weights = F.softmax(self.tag_attention(tag_embedded), dim=1)
        genre_weights = F.softmax(self.genre_attention(genre_embedded), dim=1)
        
        # Weight embeddings by their relevance scores and attention weights
        tag_embedded = torch.sum(tag_embedded * tag_weights * tag_rel.unsqueeze(-1), dim=1)
        genre_embedded = torch.sum(genre_embedded * genre_weights * genre_mask.unsqueeze(-1), dim=1) / torch.sum(genre_mask, dim=1, keepdim=True)
        
        # Concatenate embeddings
        concatenated = torch.cat((user_embedded, movie_embedded, tag_embedded, genre_embedded), dim=-1)
        concatenated = self.dropout(concatenated)
        
        # Final prediction
        x = torch.relu(self.fc(concatenated))
        return x.view(-1)


    def training_step(self, batch, batch_idx):
        user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, rating, weight = batch
        predicted_rating = self(user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask)

        # Apply weights to the loss
        loss = torch.mean(weight * (predicted_rating - rating) ** 2)

        self.log('train_loss', loss)

        # Calculate weighted MSE and MAE
        weighted_mse = torch.mean(weight * (predicted_rating - rating) ** 2)
        weighted_mae = torch.mean(weight * torch.abs(predicted_rating - rating))

        self.log('train_mse', weighted_mse)
        self.log('train_mae', weighted_mae)

        self.train_mse_scores.append(weighted_mse)
        self.train_mae_scores.append(weighted_mae)

        return loss

    def validation_step(self, batch, batch_idx):
        user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, rating, weight = batch
        predicted_rating = self(user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask)

        # Apply weights to the loss
        loss = torch.mean(weight * (predicted_rating - rating) ** 2)

        self.log('val_loss', loss)

        # Calculate weighted MSE and MAE
        weighted_mse = torch.mean(weight * (predicted_rating - rating) ** 2)
        weighted_mae = torch.mean(weight * torch.abs(predicted_rating - rating))

        self.log('val_mse', weighted_mse)
        self.log('val_mae', weighted_mae)

        self.val_mse_scores.append(weighted_mse)
        self.val_mae_scores.append(weighted_mae)

    def test_step(self, batch, batch_idx):
        user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask, rating, weight = batch
        predicted_rating = self(user_id, movie_id, tag_ids, tag_rel, genre_indices, genre_mask)

        # Apply weights to the loss
        loss = torch.mean(weight * (predicted_rating - rating) ** 2)

        self.log('test_loss', loss)

        # Calculate weighted MSE and MAE
        weighted_mse = torch.mean(weight * (predicted_rating - rating) ** 2)
        weighted_mae = torch.mean(weight * torch.abs(predicted_rating - rating))

        self.log('test_mse', weighted_mse)
        self.log('test_mae', weighted_mae)

    def on_train_epoch_end(self):
        avg_train_mse = torch.stack(self.train_mse_scores).mean()
        avg_train_mae = torch.stack(self.train_mae_scores).mean()
        self.log('avg_train_mse', avg_train_mse, prog_bar=True)
        self.log('avg_train_mae', avg_train_mae, prog_bar=True)
        self.train_mse_scores.clear()
        self.train_mae_scores.clear()
        print(f'\n[Train] MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}')

    def on_validation_epoch_end(self):
        avg_val_mse = torch.stack(self.val_mse_scores).mean()
        avg_val_mae = torch.stack(self.val_mae_scores).mean()
        self.log('avg_val_mse', avg_val_mse, prog_bar=True)
        self.log('avg_val_mae', avg_val_mae, prog_bar=True)
        self.val_mse_scores.clear()
        self.val_mae_scores.clear()
        print(f'\n[Validation] MSE: {avg_val_mse:.4f}, MAE: {avg_val_mae:.4f}')


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.optimal_params['lr'])
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.optimal_params['step_size'], gamma=self.optimal_params['gamma']),
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]
    
    
    def compute_tag_attention_weights(self, tag_ids, tag_embedded):
        tag_weights = F.softmax(self.tag_attention(tag_embedded), dim=1)
        return tag_weights

    def compute_genre_attention_weights(self, genre_embedded):
        genre_weights = F.softmax(self.genre_attention(genre_embedded), dim=1)
        return genre_weights

    def predict_ratings(self, movie_ids, user_id=None):
        # Check if the user_id is within the range of the user embeddings
        if user_id is not None and user_id.item() >= self.user_embedding.weight.size(0):
            print(f"User ID {user_id.item()} is out of range. Falling back to content-based recommendations.")
            user_id = None


        # If a user_id is provided, repeat it for each movie_id
        if user_id is not None:
            user_id = user_id.unsqueeze(0).repeat(len(movie_ids), 1).squeeze(1)

        # Retrieve tags and genres for each movie
        tag_ids, tag_rels = zip(*(get_tags_for_movie(mid.item(), self.tag_scores_df) for mid in movie_ids))
        genre_indices, genre_masks = zip(*(get_genres_for_movie(mid.item(), self.movie_genres_df) for mid in movie_ids))

        # Convert to tensors and transfer to the appropriate device
        tag_ids = torch.stack(tag_ids).to(self.device)
        tag_rels = torch.stack(tag_rels).to(self.device)
        genre_indices = torch.stack(genre_indices).to(self.device)
        genre_masks = torch.stack(genre_masks).to(self.device)

        # Calculate predicted ratings using the model's forward pass
        predicted_ratings = self.forward(movie_ids, tag_ids, tag_rels, genre_indices, genre_masks, user_id)

        # Embed tags and genres, then compute attention weights
        tag_embedded = self.tag_embedding(tag_ids)
        genre_embedded = self.genre_embedding(genre_indices)
        tag_attn_weights = self.compute_tag_attention_weights(tag_ids, tag_embedded)
        genre_attn_weights = self.compute_genre_attention_weights(genre_embedded)

        # Get top tag and genre indices
        top_tag_indices = torch.topk(tag_attn_weights, k=3, dim=1).indices + 1
        top_genre_indices = torch.topk(genre_attn_weights, k=3, dim=1).indices + 1

        # Retrieve tag and genre names
        tag_names = [self.tags_df[self.tags_df['tagId'] == idx.item()]['tag'].values[0] for idx in top_tag_indices.flatten()]
        genre_names = [self.genres_df[self.genres_df['genreId'] == idx.item()]['genreName'].values[0] for idx in top_genre_indices.flatten()]

        results = []
        # Prepare a structured output for each movie
        for i in range(len(movie_ids)):
            movie_info = {
                "movie_id": movie_ids[i].item(),
                "predicted_rating": round(float(predicted_ratings[i].item()), 1),
                "tags": tag_names[i*3:(i+1)*3],
                "genres": genre_names[i*3:(i+1)*3]
            }
            results.append(movie_info)
        
        return results
    
    def predict_rating_visual(self, movie_ids, user_id=None):
        # Check if the user_id is within the range of the user embeddings
        if user_id is not None and user_id.item() >= self.user_embedding.weight.size(0):
            print(f"User ID {user_id.item()} is out of range. Falling back to content-based recommendations.")
            user_id = None


        # If a user_id is provided, repeat it for each movie_id
        if user_id is not None:
            user_id = user_id.unsqueeze(0).repeat(len(movie_ids), 1).squeeze(1)

        # Retrieve tags and genres for each movie
        tag_ids, tag_rels = zip(*(get_tags_for_movie(mid.item(), self.tag_scores_df) for mid in movie_ids))
        genre_indices, genre_masks = zip(*(get_genres_for_movie(mid.item(), self.movie_genres_df) for mid in movie_ids))

        # Convert to tensors and transfer to the appropriate device
        tag_ids = torch.stack(tag_ids).to(self.device)
        tag_rels = torch.stack(tag_rels).to(self.device)
        genre_indices = torch.stack(genre_indices).to(self.device)
        genre_masks = torch.stack(genre_masks).to(self.device)

        # Calculate predicted ratings using the model's forward pass
        predicted_ratings = self.forward(movie_ids, tag_ids, tag_rels, genre_indices, genre_masks, user_id)

        # Embed tags and genres, then compute attention weights
        tag_embedded = self.tag_embedding(tag_ids)
        genre_embedded = self.genre_embedding(genre_indices)
        tag_attn_weights = self.compute_tag_attention_weights(tag_ids, tag_embedded)
        genre_attn_weights = self.compute_genre_attention_weights(genre_embedded)

        # Get top tag and genre indices
        top_tag_indices = torch.topk(tag_attn_weights, k=10, dim=1).indices + 1
        top_genre_indices = torch.topk(genre_attn_weights, k=10, dim=1).indices + 1

        # Retrieve tag and genre names
        tag_names = [self.tags_df[self.tags_df['tagId'] == idx.item()]['tag'].values[0] for idx in top_tag_indices.flatten()]
        genre_names = [self.genres_df[self.genres_df['genreId'] == idx.item()]['genreName'].values[0] for idx in top_genre_indices.flatten()]

        results = []
        # Prepare a structured output for each movie
        for i in range(len(movie_ids)):
            movie_info = {
                "movie_id": movie_ids[i].item(),
                "predicted_rating": round(float(predicted_ratings[i].item()), 1),
                "tags": tag_names[i*3:(i+1)*3],
                "genres": genre_names[i*3:(i+1)*3]
            }
            results.append(movie_info)
        
        return results
    


def get_tags_for_movie(movie_id, tag_scores_df, max_tags=10):
    movie_tags = tag_scores_df[tag_scores_df['movieId'] == movie_id].nlargest(max_tags, 'relevance')
    tag_ids = movie_tags['tagId'].tolist()
    tag_rel = movie_tags['relevance'].tolist()
    # Pad if necessary
    tag_ids += [0] * (max_tags - len(tag_ids))
    tag_rel += [0.0] * (max_tags - len(tag_rel))
    return torch.tensor(tag_ids, dtype=torch.long), torch.tensor(tag_rel, dtype=torch.float)


def get_genres_for_movie(movie_id, movie_genres_df, max_genres=10):
    movie_genres = movie_genres_df[movie_genres_df['movieId'] == movie_id]['genreId'].values
    movie_genres = list(movie_genres[:max_genres])  # Truncate if necessary
    movie_genres = movie_genres + [0] * (max_genres - len(movie_genres))  # Pad with zeros
    genre_mask = [1.0 if i < len(movie_genres) else 0.0 for i in range(max_genres)]  # Mask to distinguish real genres from padding
    return torch.tensor(movie_genres, dtype=torch.long), torch.tensor(genre_mask, dtype=torch.float)


def load_data(max_tags=10):
    # Load the data
    movies_df = pd.read_csv('subset_data/sampled_movies.csv')
    genres_df = pd.read_csv('subset_data/genres.csv')
    movie_genres_df = pd.read_csv('subset_data/sampled_movie_genres.csv')
    ratings_df = pd.read_csv('subset_data/sampled_ratings.csv')
    tag_scores = pd.read_csv('subset_data/sampled_tag_scores.csv')
    tags_df = pd.read_csv('subset_data/tags.csv')

    # Merge tag_scores with tags_df to bring in the 'tag' column
    tag_scores = tag_scores.merge(tags_df, on='tagId', how='left')


    return movies_df, genres_df, movie_genres_df, ratings_df, tag_scores, tags_df