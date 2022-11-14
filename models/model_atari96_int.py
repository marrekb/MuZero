import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.init import init_orthogonal_head, init_orthogonal_features
from utils.model import ResidualBlock, normalize_output

class ValueModel(nn.Module):
    def __init__(self, value_ext_support, value_int_support):
        super(ValueModel, self).__init__()

        self.representation_model = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3,
                      stride = 2, padding = 1),
            nn.ReLU(),
            ResidualBlock(96),
            nn.MaxPool2d(2, 2),
            ResidualBlock(96),
            nn.AvgPool2d(2, 2)
        )
        self.representation_model.apply(init_orthogonal_features)

        self.prediction_fc = nn.Sequential(
            nn.Flatten(),
             nn.Linear(3456, 512),
             nn.ReLU()
        )
        self.prediction_fc.apply(init_orthogonal_features)

        self.value_ext_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, value_ext_support * 2 + 1)
        )
        self.value_ext_model.apply(init_orthogonal_head)

        self.value_int_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, value_int_support * 2 + 1)
        )
        self.value_int_model.apply(init_orthogonal_head)

    def forward(self, observations):
        states = self.representation_model(observations)
        x, ext_values, int_values = self.recurrent_inference(states)
        return states, x, ext_values, int_values

    def recurrent_inference(self, states):
        x = self.prediction_fc(states)
        return x, self.value_ext_model(x), self.value_int_model(x)

class Model(nn.Module):
    def __init__(self, count_of_actions, value_ext_support, 
                 value_int_support, reward_support):
        super(Model, self).__init__()

        self.value_model = ValueModel(value_ext_support, value_int_support)

        self.dynamic_model = nn.Sequential(
            nn.Conv2d(97, 96, 3, stride = 1, padding = 1),
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96)
        )
        self.dynamic_model.apply(init_orthogonal_features)

        self.reward_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3456, 512),
            nn.ReLU()
        )
        self.reward_model.apply(init_orthogonal_features)

        self.reward_ext_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, reward_support * 2 + 1)
        )
        self.reward_ext_model.apply(init_orthogonal_head)

        self.reward_int_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, reward_support * 2 + 1)
        )
        self.reward_int_model.apply(init_orthogonal_head)

        self.policy_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, count_of_actions)
        )
        self.policy_model.apply(init_orthogonal_head)

    def initial_inference(self, observations):
        states, x, ext_values, int_values = self.value_model(observations)
        return states, self.policy_model(x), ext_values, int_values

    def recurrent_inference(self, states, actions):
        states = normalize_output(states)
        f_states = torch.cat((states, actions), dim = 1)
        new_states = self.dynamic_model(f_states)
        rewards = self.reward_model(new_states)
        rewards_ext = self.reward_ext_model(rewards)
        rewards_int = self.reward_int_model(rewards)
        x, ext_values, int_values = self.value_model.recurrent_inference(new_states)
        return new_states, rewards_ext, rewards_int, self.policy_model(x), ext_values, int_values
