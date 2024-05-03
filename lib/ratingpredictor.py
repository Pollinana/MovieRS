import torch
import torch.nn as nn

class RatingPredictor(nn.Module):
    def __init__(self, num_users=943, num_movies=1862, user_embedding_dim=32, movie_embedding_dim=32, 
                 user_feature_dim=3, movie_feature_dim=18, nhead=5,num_encoder_layers=1,dim_feedforward=1024, dropout_rate=1):
        super(RatingPredictor, self).__init__()

        self.user_embedding = nn.Embedding(num_embeddings=num_users + 1, embedding_dim=user_embedding_dim)
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies + 1, embedding_dim=movie_embedding_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=user_embedding_dim + movie_embedding_dim + user_feature_dim + movie_feature_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers= num_encoder_layers)

        self.fc = nn.Linear(user_embedding_dim + movie_embedding_dim + user_feature_dim + movie_feature_dim, 1)

    def forward(self, user_ids, movie_ids, user_features, movie_features):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        # 用户ID的Embedding与用户特征拼接
        user_combined_features = torch.cat([user_embed, user_features], dim=-1)
        # 电影ID的Embedding与电影特征拼接
        movie_combined_features = torch.cat([movie_embed, movie_features], dim=-1)
        
        # 将用户和电影的组合特征拼接
        combined_features = torch.cat([user_combined_features, movie_combined_features], dim=-1).unsqueeze(1)  # 增加一个维度以符合Transformer输入要求
        
        # 通过Transformer Encoder
        transformer_output = self.transformer_encoder(combined_features)
        transformer_output = transformer_output.squeeze(1)  # 移除多余的维度
        
        # 通过全连接层
        output = self.fc(transformer_output)

        return output.squeeze()