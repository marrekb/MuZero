import torch
import torch.nn.functional as F
import numpy as np
from utils.model import support_to_scalar, normalize_output

class MCTS:
    def __init__(self, count_of_actions, filters = 256, features_dim = (6, 6),
                 count_of_simulations = 50, c1 = 1.25, c2 = 19652.0, T = 1.0,
                 dirichlet_alpha = 0.25, exploration_fraction = 0.25,
                 gamma_ext = 0.997, gamma_int = 0.99, device = 'cpu',
                 ext_coef = 2.0, int_coef = 1.0):

        self.count_of_actions = count_of_actions
        self.state_dim = (filters,) + features_dim
        self.action_dim = (1, ) + features_dim

        self.count_of_nodes = count_of_simulations + 1
        self.count_of_simulations = count_of_simulations

        self.c1 = c1
        self.c2 = c2
        self.gamma_ext = gamma_ext
        self.gamma_int = gamma_int

        self.ext_coef = ext_coef
        self.int_coef = int_coef

        self.sum_coef = ext_coef + int_coef

        self.T = 1.0

        self.dirichlet_vector = np.ones(self.count_of_actions) * dirichlet_alpha
        self.exploration_fraction = exploration_fraction
        self.exploration_fraction_inv = 1 - self.exploration_fraction

        self.device = device

        self.action_t = torch.zeros((count_of_actions,) + self.action_dim, device = device)
        self.order_actions = torch.arange(count_of_actions, device = device)
        action_values = self.order_actions / float(count_of_actions)
        self.action_t += action_values.view(-1, 1, 1, 1)

    def run(self, observations, model, training = False, value_ext_support = 3, value_int_support = 3, 
            reward_support = 1):
        # creating tensor structure of tree
        count_of_obs = len(observations)
        length = count_of_obs * self.count_of_nodes + 1
        t_S = torch.zeros(((length,) + self.state_dim), device = self.device)
        t_P = torch.zeros((length, self.count_of_actions), device = self.device)

        t_Q_ext = torch.zeros((length, self.count_of_actions), device = self.device)
        t_Q_int = torch.zeros((length, self.count_of_actions), device = self.device)

        t_R_ext = torch.zeros((length, self.count_of_actions), device = self.device)
        t_R_int = torch.zeros((length, self.count_of_actions), device = self.device)

        t_N = torch.zeros((length, self.count_of_actions), device = self.device, dtype = torch.int32)
        t_C = torch.zeros((length, self.count_of_actions), device = self.device, dtype = torch.long)

        # creating root nodes
        order = torch.arange(count_of_obs, device = self.device)
        root_indices = order + 1
        states, policies, root_ext_values, root_int_values = model.initial_inference(observations)
        root_ext_values = support_to_scalar(root_ext_values, value_ext_support)
        root_int_values = support_to_scalar(root_int_values, value_int_support)

        t_S[root_indices] = states
        t_P[root_indices] = F.softmax(policies, dim = 1)

        # visiting each action in root nodes
        actions_t = self.action_t.repeat(count_of_obs, 1, 1, 1)
        root_action_indices = root_indices.repeat_interleave(self.count_of_actions)
        action_indices = self.order_actions.repeat(count_of_obs)

        states, rewards_ext, rewards_int, policies, values_ext, values_int = model.recurrent_inference(t_S[root_action_indices], actions_t)
        policies = F.softmax(policies, dim = 1)
        rewards_ext = support_to_scalar(rewards_ext, reward_support).view(-1)
        rewards_int = support_to_scalar(rewards_int, reward_support).view(-1)

        values_ext = support_to_scalar(values_ext, value_ext_support).view(-1)
        values_int = support_to_scalar(values_int, value_int_support).view(-1)

        #rewards, values = rewards.view(-1), values.view(-1)
        order_actions = torch.arange(count_of_obs * self.count_of_actions, device = self.device)
        indices = order_actions + count_of_obs + 1

        t_S[indices] = states #normalize_output(states)
        t_P[indices] = policies
        t_R_ext[root_action_indices, action_indices] = rewards_ext
        t_R_int[root_action_indices, action_indices] = rewards_int

        t_Q_ext[root_action_indices, action_indices] = rewards_ext + self.gamma_ext * values_ext
        t_Q_int[root_action_indices, action_indices] = rewards_int + self.gamma_int * values_int

        t_N[root_indices] = 1
        t_C[root_action_indices, action_indices] = indices

        # allocating memory for simulations
        mem_actions = torch.zeros((length, count_of_obs), device = self.device, dtype = torch.long)
        mem_last_step = torch.zeros(count_of_obs, device = self.device, dtype = torch.long)

        # executing simulations
        for simulation in range(self.count_of_actions, self.count_of_simulations):
            mem_indices = torch.zeros((length, count_of_obs), device = self.device, dtype = torch.long)
            mem_indices[0] = root_indices

            # phase of selection
            step = 0
            selection = True

            while selection:
                indices = torch.nonzero(mem_indices[step])
                dim0, _ = indices.shape
                if dim0 > 0:
                    indices = indices.view(-1)
                    node_indices = mem_indices[step, indices]
                    n_sum = torch.sum(t_N[node_indices].float(), dim = 1).view(-1, 1)
                    n_sqrt_sum = torch.sqrt(n_sum + 1).view(-1, 1)
                    c = self.c1 + torch.log((n_sum + self.c2 + 1) / self.c2)

                    q = (self.ext_coef * t_Q_ext[node_indices] + self.int_coef * t_Q_int[node_indices]) / self.sum_coef
                    q_min = torch.min(q, dim = 1).values.view(-1, 1)
                    q_max = torch.max(q, dim = 1).values.view(-1, 1)
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
            actions_t += (actions / float(self.count_of_actions)).view(-1, 1, 1, 1)

            new_states, rewards_ext, rewards_int, policies, values_ext, values_int = model.recurrent_inference(states, actions_t)
            policies = F.softmax(policies, dim = 1)
            #values = values.view(-1)
            rewards_ext = support_to_scalar(rewards_ext, reward_support).view(-1)
            rewards_int = support_to_scalar(rewards_int, reward_support).view(-1)

            values_ext = support_to_scalar(values_ext, value_ext_support).view(-1)
            values_int = support_to_scalar(values_int, value_int_support).view(-1)

            # phase of expansion
            new_indices = root_indices + ((simulation + 1) * count_of_obs)

            t_S[new_indices] = new_states #normalize_output(new_states)
            t_P[new_indices] = policies
            t_R_ext[last_indices, actions] = rewards_ext
            t_R_int[last_indices, actions] = rewards_int
            t_C[last_indices, actions] = new_indices

            # phase of back propagation
            for step in reversed(range(step)):
                indices = torch.nonzero(mem_indices[step]).view(-1)
                node_indices = mem_indices[step, indices]
                node_actions = mem_actions[step, indices]
                values_ext[indices] = t_R_ext[node_indices, node_actions] + self.gamma_ext * values_ext[indices]
                values_int[indices] = t_R_int[node_indices, node_actions] + self.gamma_int * values_int[indices]

                n = t_N[node_indices, node_actions]
                q_ext = t_Q_ext[node_indices, node_actions]
                q_int = t_Q_int[node_indices, node_actions]

                q_n_ext = q_ext * n
                q_n_int = q_int * n

                n += 1
                t_Q_ext[node_indices, node_actions] = (q_n_ext + values_ext[indices]) / n
                t_Q_int[node_indices, node_actions] = (q_n_int + values_int[indices]) / n

                t_N[node_indices, node_actions] = n

        n = t_N[root_indices]**self.T
        n_sum = torch.sum(n, dim = 1).float().view(-1, 1)
        root_policies = (n / n_sum)

        if training:
            actions = root_policies.multinomial(num_samples = 1)
        else:
            actions = torch.argmax(root_policies, dim = 1).view(-1, 1)
        return actions.cpu(), root_policies.cpu(), root_ext_values.cpu(), root_int_values.cpu()
