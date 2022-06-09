import numpy as np
import torch
from methods.mcts import MCTS, Node

class MuZero:
    def __init__(self, count_of_actions, filters = 64, features_dim = (6, 6), device = 'cpu',
                 count_of_simulations = 50, c1 = 1.25, c2 = 19652.0,
                 gamma = 0.997,  dirichlet_alpha = 0.25, exploration_fraction = 0.25):
        self.count_of_actions = count_of_actions
        self.state_dim = (filters,) + features_dim
        self.action_dim = (count_of_actions,) + features_dim
        self.device = device

        self.count_of_simulations = count_of_simulations
        self.length_of_mcts = count_of_simulations + 1

        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma

        self.dirichlet_vector = np.ones(self.count_of_actions) * dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.exploration_fraction_inv = 1 - self.exploration_fraction

    def mcts(self, observations, model,  predict_from_observations,
             predict_from_states, training = False):
        states, policies, values = predict_from_observations(observations, model)
        count_of_obs = len(values)
        order = torch.arange(count_of_obs, device = self.device)

        if training:
            noise = torch.from_numpy(np.random.dirichlet(self.dirichlet_vector, count_of_obs)).to(self.device).float()
            policies = policies * self.exploration_fraction_inv + noise * self.exploration_fraction

        root_list = [Node(states[i], policies[i], self.device) for i in range(count_of_obs)]
        mcts_list = [MCTS(root, self.gamma, self.c1, self.c2) for root in root_list]

        for simulation in range(self.count_of_simulations):
            combinations = [mcts.selection() for mcts in mcts_list]
            states, actions = map(list, zip(*combinations))
            states, actions = torch.stack(states).to(self.device), torch.tensor(actions).to(self.device)
            actions_t = torch.zeros(((count_of_obs,) + self.action_dim), device = self.device)
            actions_t[order, actions] = 1

            new_states, rewards, policies, values = predict_from_states(states, actions_t, model)

            for i in range(count_of_obs):
                new_node = Node(new_states[i], policies[i], self.device)
                parent, action = mcts_list[i].parents[0]
                parent.C[action] = new_node
                parent.R[action] = rewards[i, 0]
                mcts_list[i].backup(values[i, 0])

        root_data = [root.getRootData() for root in root_list]
        root_policies, root_values = map(list, zip(*root_data))
        root_policies, root_values = torch.stack(root_policies).cpu(), torch.stack(root_values).cpu()

        if training:
            actions = root_policies.multinomial(num_samples = 1)
        else:
            actions = torch.argmax(root_policies, dim = 1).view(-1, 1)
        return actions, root_policies, root_values
