import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    
    def __init__(
        self, 
        num_users, 
        num_movies,
        embedding_dim=32
    ):
        super().__init__()
        
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_dim
        )
        
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_dim
        )
        
        self.user_bias = nn.Embedding(
            num_users,
            1
        )
        
        self.movie_bias = nn.Embedding(
            num_movies,
            1
        )
        
        self.global_bias = nn.Parameter(
            torch.zeros(1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        
        nn.init.normal_(
            self.user_embedding.weight,
            std=0.01
        )
        
        nn.init.normal_(
            self.movie_embedding.weight,
            std=0.01
        )
        
        nn.init.zeros_(
            self.user_bias.weight
        )
        
        nn.init.zeros_(
            self.movie_bias.weight
        )
        
    def forward(
        self,
        user_ids,
        movie_ids
    ):
        
        user_vector = self.user_embedding(
            user_ids
        )
        
        movie_vector = self.movie_embedding(
            movie_ids
        )
        
        interaction = (
            user_vector * movie_vector
        ).sum(dim=1)
        
        user_bias = self.user_bias(
            user_ids
        ).squeeze(1)
        
        movie_bias = self.movie_bias(
            movie_ids
        ).squeeze(1)
        
        prediction = (
            self.global_bias
            + user_bias
            + movie_bias
            + interaction
        )
        
        return prediction