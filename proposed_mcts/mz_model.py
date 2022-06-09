import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def weights_init_orthogonal_head(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(layer.weight, 0.01)
        init.zeros_(layer.bias)

def weights_init_orthogonal_features(layer):
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(layer.weight, 2**0.5)
        init.zeros_(layer.bias)

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(filters, filters, 3, stride = 1, padding = 1)

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = x + input
        return F.relu(x)

class Model(nn.Module):
    def __init__(self, count_of_actions, features_size):
        super(Model, self).__init__()
        self.features_model = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride = 2, padding = 1),
            nn.ReLU(),
            ResidualBlock(128),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride = 2, padding = 1),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.AvgPool2d(2, stride=2),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.AvgPool2d(2, stride=2)
        )
        self.features_model.apply(weights_init_orthogonal_features)

        self.fc_f = nn.Linear(features_size, 512)
        self.fc_f.apply(weights_init_orthogonal_features)
        self.features_size = features_size

        self.dynamic_model = nn.Sequential(
            nn.Conv2d(257, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        self.dynamic_model.apply(weights_init_orthogonal_features)

        self.policy_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, count_of_actions)
        )
        self.policy_model.apply(weights_init_orthogonal_head)

        self.value_model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.value_model.apply(weights_init_orthogonal_head)

        self.reward_conv = nn.Conv2d(256, 256, 3, stride = 2, padding = 1)
        self.reward_conv.apply(weights_init_orthogonal_features)

        self.reward_features_size = features_size // 4
        self.reward_model = nn.Sequential(
            nn.Linear(self.reward_features_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.reward_model.apply(weights_init_orthogonal_head)

    def presentation_function(self, input):
        return self.features_model(input)

    def predict_function(self, input):
        x = F.relu(self.fc_f(input.view(-1, self.features_size)))
        return self.policy_model(x), self.value_model(x)

    def dynamic_function(self, input):
        x = self.dynamic_model(input)
        xr = F.relu(self.reward_conv(x)).view(-1, self.reward_features_size)
        return x, self.reward_model(xr)

    def forward(self, input):
        x = self.presentation_function(input)
        policy, value = self.predict_function(x)
        return x, policy, value
