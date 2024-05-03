import torch 
import torch.nn as nn
from collections import namedtuple, deque
import random
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """存储一次交互经历"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """随机采样"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class TransformerNetV2(nn.Module):
    def __init__(self, user_size, movie_size, embed_size, num_heads, num_layers, num_actions):
        super(TransformerNetV2, self).__init__()
        self.user_embed = nn.Embedding(num_embeddings=user_size, embedding_dim=embed_size)
        self.movie_embed = nn.Embedding(num_embeddings=movie_size, embedding_dim=embed_size)      
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size+embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_layers)
        
        # 输出层，从Transformer平均后的输出转换到动作空间大小
        self.fc_out = nn.Linear(embed_size+embed_size, num_actions)
        
    def forward(self, state):
        # 手动增加批次维度（batch dimension）
#         state = state.unsqueeze(0)  # state的形状从[6]变为[1, 6]，模拟批次大小为1

        user_ids = state[:, 0]
        movie_ids = state[:, 1:6]

        # 嵌入
        user_embeddings = self.user_embed(user_ids)
        movie_embeddings = self.movie_embed(movie_ids).view(*movie_ids.shape, -1).mean(dim=1)

        # 将用户嵌入与电影嵌入合并，得到[1, 6, embed_size]的张量
        embeddings = torch.cat((user_embeddings, movie_embeddings), dim=1)

        # Transformer处理
        transformer_output = self.transformer_encoder(embeddings)
        # 将输出转化为[1, -1]的形状以匹配全连接层的期望输入
        transformer_output = transformer_output.view(-1, self.fc_out.in_features)

        # 输出动作的Q值
        action_q_values = self.fc_out(transformer_output)
        return action_q_values
    
class DoubleDQN:
    def __init__(self, policy_net, target_net, memory, gamma, epsilon, optimizer, action_size, device):
        self.policy_net = policy_net
        self.target_net = target_net
        self.memory = memory
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.action_size = action_size
        self.device = device
        # 将目标网络的参数同步到策略网络
        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()  # 目标网络处于评估模式 
        self.TARGET_REPLACE_ITER = 1000
        self.learn_step_counter = 0
        self.step_counter = 0


    def select_action(self, state):
        sample = random.random()

        #epsilon greedy
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)

    def optimize_model(self, batch_size, capacity):
        self.step_counter += 1
        if self.step_counter < 0.8 * capacity:   
            return
        self.step_counter = 0
        self.learn_step_counter += 1
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict()) 

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # 计算非最终状态的掩码并连接batch元素
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(self.device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # 计算Q值
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size).to(self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 计算Huber损失
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()