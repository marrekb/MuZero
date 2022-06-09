import numpy as np
import torch

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

    '''
        predict_from_observations and predict_from_states are pointers to the
        prediction methods of deep neural network
    '''
    def mcts(self, observations, model, predict_from_observations,
             predict_from_states, training = False):

        # creating tensor structure of tree
        count_of_obs = len(observations)
        length = count_of_obs * self.length_of_mcts + 1
        t_S = torch.zeros(((length,) + self.state_dim), device = self.device)
        t_P = torch.zeros((length, self.count_of_actions), device = self.device)
        t_Q = torch.zeros((length, self.count_of_actions), device = self.device)
        t_R = torch.zeros((length, self.count_of_actions), device = self.device)
        t_N = torch.zeros((length, self.count_of_actions), device = self.device, dtype = torch.int32)
        t_C = torch.zeros((length, self.count_of_actions), device = self.device, dtype = torch.long)

        # creating root nodes
        order = torch.arange(count_of_obs, device = self.device)
        root_indices = order + 1

        states, policies, values = predict_from_observations(observations, model)
        t_S[root_indices] = states

        # adding Dirichlet noise (unfortunatelly, there is no PyTorch implementation of Dirichlet noise)
        if training:
            noise = torch.from_numpy(np.random.dirichlet(self.dirichlet_vector, count_of_obs)).to(self.device).float()
            new_policies = policies * self.exploration_fraction_inv + noise * self.exploration_fraction
            t_P[root_indices] = new_policies
        else:
            t_P[root_indices] = policies

        # taking each action in root nodes
        action_t = torch.zeros((self.count_of_actions,) + self.action_dim, device = self.device)
        order_actions = torch.arange(self.count_of_actions, device = self.device)
        action_t[order_actions, order_actions] = 1
        actions_t = action_t.repeat(count_of_obs, 1, 1, 1)
        root_action_indices = root_indices.repeat_interleave(self.count_of_actions)
        action_indices = order_actions.repeat(count_of_obs)

        # allocating memory for simulations
        mem_actions = torch.zeros((self.length_of_mcts, count_of_obs), device = self.device, dtype = torch.long)
        mem_last_step = torch.zeros(count_of_obs, device = self.device, dtype = torch.long)

        # executing simulations
        for simulation in range(0, self.count_of_simulations):
            mem_indices = torch.zeros((self.length_of_mcts, count_of_obs), device = self.device, dtype = torch.long)
            mem_indices[0] = root_indices

            # phase of selection
            step = 0
            selection = True

            q_min, q_max = t_Q.min(), t_Q.max()
            while selection:
                indices = torch.nonzero(mem_indices[step])
                dim0, dim1 = indices.shape
                if dim0 > 0:
                    indices = indices.view(-1)
                    node_indices = mem_indices[step, indices]
                    n_sum = torch.sum(t_N[node_indices].float(), dim = 1).view(-1, 1)
                    n_sqrt_sum = torch.sqrt(n_sum + 1).view(-1, 1)
                    c = self.c1 + torch.log((n_sum + self.c2 + 1) / self.c2)
                    q = t_Q[node_indices]
                    norm_q = (q - q_min) / (q_max - q_min + 0.0001)
                    u = norm_q + t_P[node_indices] * ((n_sqrt_sum) / (1 + t_N[node_indices])) * c
                    u_indices = torch.argmax(u, dim = 1)
                    mem_actions[step, indices] = u_indices
                    mem_last_step[indices] = step
                    step += 1
                    mem_indices[step, indices] = t_C[node_indices, u_indices]
                else:
                    selection = False

            # phase of simulation
            last_indices = mem_indices[mem_last_step, order]
            states = t_S[last_indices]
            actions = mem_actions[mem_last_step, order]
            actions_t = torch.zeros(((count_of_obs,) + self.action_dim), device = self.device)
            actions_t[order, actions] = 1

            new_states, rewards, policies, values = predict_from_states(states, actions_t, model)
            values = values.view(-1)
            # phase of expansion
            new_indices = root_indices + ((simulation + 1) * count_of_obs)

            t_S[new_indices] = new_states
            t_P[new_indices] = policies
            t_R[last_indices, actions] = rewards.view(-1)
            t_C[last_indices, actions] = new_indices

            # phase of back propagation
            for step in reversed(range(step)):
                indices = torch.nonzero(mem_indices[step]).view(-1)
                node_indices = mem_indices[step, indices]
                node_actions = mem_actions[step, indices]
                values[indices] = t_R[node_indices, node_actions] + self.gamma * values[indices]
                q = t_Q[node_indices, node_actions]
                n = t_N[node_indices, node_actions]
                q_n = q * n
                n += 1

                t_Q[node_indices, node_actions] = (q_n + values[indices]) / n
                t_N[node_indices, node_actions] = n

        n = t_N[root_indices]
        n_sum = torch.sum(n, dim = 1).float().view(-1, 1)
        root_policies = (n / n_sum)
        root_values = torch.sum(t_Q[root_indices] * root_policies, dim = 1).view(-1, 1).cpu()

        if training:
            actions = root_policies.multinomial(num_samples = 1)
        else:
            actions = torch.argmax(root_policies, dim = 1).view(-1, 1)
        return actions, root_policies, root_values
