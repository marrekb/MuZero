import numpy as np
import torch
from torch.multiprocessing import Process, Pipe
from methods.mcts import MCTS, Node

def worker(connection, count_of_obs, training, states, policies, device,
           dirichlet_vector, count_of_simulations, exploration_fraction_inv,
           exploration_fraction, gamma, c1, c2, action_dim):
    order = torch.arange(count_of_obs, device = device)

    if training:
        noise = torch.from_numpy(np.random.dirichlet(dirichlet_vector, count_of_obs)).to(device).float()
        policies = policies * exploration_fraction_inv + noise * exploration_fraction

    root_list = [Node(states[i], policies[i], device) for i in range(count_of_obs)]
    mcts_list = [MCTS(root, gamma, c1, c2) for root in root_list]

    for simulation in range(count_of_simulations):
        combinations = [mcts.selection() for mcts in mcts_list]
        states, actions = map(list, zip(*combinations))
        states, actions = torch.stack(states), torch.tensor(actions)
        connection.send([states, actions])
        new_states, rewards, policies, values = connection.recv()

        for i in range(count_of_obs):
            new_node = Node(new_states[i], policies[i], device)
            parent, action = mcts_list[i].parents[0]
            parent.C[action] = new_node
            parent.R[action] = rewards[i, 0]
            mcts_list[i].backup(values[i, 0])

    root_data = [root.getRootData() for root in root_list]
    root_policies, root_values = map(list, zip(*root_data))
    connection.send([torch.stack(root_policies), torch.stack(root_values)])
    connection.recv()
    connection.close()

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

    def mcts(self, observations, model, predict_from_observations,
             predict_from_states, training = False, cpu = 10):
        states, policies, values = predict_from_observations(observations, model)
        count = len(values)
        count_of_obs = count // cpu

        states, policies = states.view((-1, count_of_obs,) + self.state_dim), policies.view(-1, count_of_obs, self.count_of_actions)
        processes, connections = [], []
        for i in range(cpu):
            parrent_connection, child_connection = Pipe()
            process = Process(target = worker, args = (child_connection, count_of_obs, training, states[i].cpu(), policies[i].cpu(), 'cpu', self.dirichlet_vector,
                                                       self.count_of_simulations, self.exploration_fraction_inv, self.exploration_fraction, self.gamma, self.c1, self.c2, self.action_dim))
            connections.append(parrent_connection)
            processes.append(process)
            process.start()

        order = torch.arange(count, device = self.device)

        for simulation in range(self.count_of_simulations):
            states, actions = [], []
            for i in range(cpu):
                s, a = connections[i].recv()
                states.append(s)
                actions.append(a)
            states, actions = torch.stack(states).to(self.device).view((-1,) + self.state_dim), torch.stack(actions).to(self.device).view(-1)

            actions_t = torch.zeros(((count,) + self.action_dim), device = self.device)
            actions_t[order, actions] = 1

            new_states, rewards, policies, values = predict_from_states(states, actions_t, model)
            new_states, rewards = new_states.view((-1, count_of_obs,) + self.state_dim), rewards.view(-1, count_of_obs, 1)
            policies, values = policies.view(-1, count_of_obs, self.count_of_actions), values.view(-1, count_of_obs, 1)

            for conn_idx in range(cpu):
                connections[conn_idx].send([new_states[conn_idx].cpu(), rewards[conn_idx].cpu(), policies[conn_idx].cpu(), values[conn_idx].cpu()])

        root_policies, root_values = [], []
        for i in range(cpu):
            p, v = connections[i].recv()
            root_policies.append(p)
            root_values.append(v)
        root_policies, root_values = torch.stack(root_policies).view(-1, self.count_of_actions), torch.stack(root_values).view(-1, 1)

        for connection in connections:
            connection.send(1)
        [process.join() for process in processes]

        if training:
            actions = root_policies.multinomial(num_samples = 1)
        else:
            actions = torch.argmax(root_policies, dim = 1).view(-1, 1)
        return actions, root_policies, root_values
