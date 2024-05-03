from .ratingpredictor import  RatingPredictor
import torch
import numpy as np
from math import exp

def prepare_states(states):
    """
    DataFrame -> states
    """
    state_values = states.values  # 获取numpy数组
    state_tensor = torch.LongTensor(state_values)  # 转换为LongTensor
    return state_tensor.unsqueeze(0)

class RecommendationEnvironment:
    def __init__(self, device, users, movies, states, checkpoint='./rating_regressor_499.pth'):
        model = RatingPredictor()
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
        self.model = model
        self.device = device
        self.users = users
        self.movies = movies
        self.prepared_states = prepare_states(states)

    def step(self, action, state):
        next_state = state.squeeze(0).tolist()
        current_movie_ids = next_state[1:6]  # 当前状态的电影ID
        next_state[1:6] = np.roll(next_state[1:6], -1) # 左移
        next_state[5] = action.item()  # 加入新电影id到最右端
        next_state = torch.LongTensor(next_state).unsqueeze(0)
        # 统计重复率
        if action.item() in current_movie_ids:
            repeat_rate = current_movie_ids.count(action.item())/5
        else:
            repeat_rate = 0
       
        if action.item() == 1682:
            movie_rating = 0
            reward = -0.5
        else:
            model_input = self.search_merge(state, action).unsqueeze(0)
            movie_rating = self.rating_model(model_input)
            if repeat_rate == 0:
                reward = exp(movie_rating-3.5)
            else:
                reward = -1 + 0.05*movie_rating-repeat_rate
            
            #TODO: finish better reward
            #reward = movie_rating + ... + ...
            # reward = exp(movie_rating-4) /(exp(2+repeat_rate) - exp(2)+0.1)-8
            
            #TODO: return info {'rating:', 'diversity': }
        return next_state, reward,{'rating': movie_rating, 'repeatRate': repeat_rate}    
    
    def rating_model(self, rinput):
        self.model.eval()
        with torch.no_grad():  # 禁用梯度计算
            for features in rinput:
                user_id = features[:,0].long().to(self.device)  # 用户ID
                movie_id = features[:,4].long().to(self.device)  # 电影ID
                user_feat = features[:,1:4].to(self.device)  # 用户特征
                movie_feat = features[:,5:].to(self.device)  # 电影特征
                prediction = self.model(user_id, movie_id, user_feat, movie_feat)
                prediction_clipped = prediction.clamp(min=1, max=5)
                rate = prediction_clipped.item()
        return rate

    def search_merge(self, state,action):
        movie_id = action.item()
        user_id = state.squeeze(0).tolist()[0]
        user_info = self.users.iloc[[user_id],:].values[0].tolist()
        movie_info = self.movies.iloc[[movie_id],:].values[0].tolist()
        merged_data = user_info + movie_info
        rate_pred_input = torch.tensor(merged_data).int()
        rate_pred_input = rate_pred_input.unsqueeze(0) 
        return rate_pred_input

if __name__ == '__main__':
    pass
    # from lib.recenv import RecommendationEnvironment
    # env = RecommendationEnvironment(device, users, movies)
    # action = torch.tensor(1).unsqueeze(0).unsqueeze(0).to(device)
    # s_, r = env.step(action, prepared_states[:,0])
    # print(s_, r)
