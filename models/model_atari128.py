import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init import init_orthogonal_head, init_orthogonal_features
from utils.model import ResidualBlock, normalize_output

class ValueModel(nn.Module):
    def __init__(self, value_support):
        super(ValueModel, self).__init__()

        self.representation_model = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),
            ResidualBlock(128),
            nn.AvgPool2d(2, 2)
        )
        self.representation_model.apply(init_orthogonal_features)

        self.prediction_fc = nn.Sequential(
            nn.Flatten(),
             nn.Linear(4608, 512),
             nn.ReLU()
        )
        self.prediction_fc.apply(init_orthogonal_features)

        self.value_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, value_support * 2 + 1)
        )
        self.value_model.apply(init_orthogonal_head)

    def forward(self, observations):
        states = self.representation_model(observations)
        x, values = self.recurrent_inference(states)
        return states, x, values

    def recurrent_inference(self, states):
        x = self.prediction_fc(states)
        return x, self.value_model(x)

class Model(nn.Module):
    def __init__(self, count_of_actions, value_support, reward_support):
        super(Model, self).__init__()

        self.value_model = ValueModel(value_support)

        self.dynamic_model = nn.Sequential(
            nn.Conv2d(129, 128, 3, stride = 1, padding = 1),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.dynamic_model.apply(init_orthogonal_features)

        self.reward_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, reward_support * 2 + 1)
        )
        self.reward_model.apply(init_orthogonal_head)

        self.policy_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, count_of_actions)
        )
        self.policy_model.apply(init_orthogonal_head)

    def initial_inference(self, observations):
        states, x, values = self.value_model(observations)
        return states, self.policy_model(x), values

    def recurrent_inference(self, states, actions):
        states = normalize_output(states)
        f_states = torch.cat((states, actions), dim = 1)
        new_states = self.dynamic_model(f_states)
        rewards = self.reward_model(new_states)
        x, values = self.value_model.recurrent_inference(new_states)
        return new_states, rewards, self.policy_model(x), values
