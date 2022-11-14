import torch
from utils.model import support_to_scalar

class ExperienceReplay:
    def __init__(self, buffer_size, batch_size, count_of_actions,
                 input_dimension, training_depth = 5, td_steps = 5,
                 gamma = 0.997, device = 'cpu'):
        self.max_size = buffer_size * batch_size
        self.batch_size = batch_size

        self.states = torch.zeros((self.max_size, *input_dimension))
        self.actions = torch.zeros((self.max_size, 1), dtype = torch.long)
        self.rewards = torch.zeros((self.max_size, 1))
        self.non_terminals = torch.zeros(self.max_size, dtype = torch.int)

        self.policies = torch.zeros((self.max_size, count_of_actions))

        self.td_values = torch.zeros((self.max_size, 1))
        self.td_rewards = torch.zeros((self.max_size, 1))
        self.td_non_terminals = torch.ones((self.max_size, 1))

        self.order_next = torch.arange(self.max_size).long() + 1
        self.order_next[self.max_size - 1] = 0

        self.td_next = torch.arange(self.max_size).long() + td_steps
        self.td_next[-td_steps:] = torch.arange(td_steps).long()

        self.td_steps = td_steps

        self.index = 0
        self.full_buffer = False
        self.K = training_depth
        self.K_ratio = 1.0 / training_depth
        
        self.gamma = gamma
        self.td_gamma = gamma**td_steps

        self.device = device

    def get_batch(self):
        count = self.max_size if self.full_buffer else self.index
        batch = torch.randint(0, count, (self.batch_size,))
        return batch

    def store(self, count, states, actions, rewards, non_terminals, policies,
              td_rewards, td_non_terminals):
        new_index = self.index + count

        if new_index <= self.max_size:
            self.states[self.index:new_index] = states
            self.actions[self.index:new_index] = actions
            self.rewards[self.index:new_index] = rewards
            self.non_terminals[self.index:new_index] = non_terminals
            self.policies[self.index:new_index] = policies
            self.td_rewards[self.index:new_index] = td_rewards
            self.td_non_terminals[self.index:new_index] = td_non_terminals

            self.index = new_index
            if self.index == self.max_size:
                self.index = 0
                self.full_buffer = True
        else:
            to_end = self.max_size - self.index
            from_start = count - to_end

            self.states[self.index:self.max_size] = states[0:to_end]
            self.actions[self.index:self.max_size] = actions[0:to_end]
            self.rewards[self.index:self.max_size] = rewards[0:to_end]
            self.non_terminals[self.index:self.max_size] = non_terminals[0:to_end]
            self.policies[self.index:self.max_size] = policies[0:to_end]
            self.td_rewards[self.index:self.max_size] = td_rewards[0:to_end]
            self.td_non_terminals[self.index:self.max_size] = td_non_terminals[0:to_end]

            self.states[0:from_start] = states[to_end:count]
            self.actions[0:from_start] = actions[to_end:count]
            self.rewards[0:from_start] = rewards[to_end:count]
            self.non_terminals[0:from_start] = non_terminals[to_end:count]
            self.policies[0:from_start] = policies[to_end:count]
            self.td_rewards[0:from_start] = td_rewards[to_end:count]
            self.td_non_terminals[0:from_start] = td_non_terminals[to_end:count]

            self.index = from_start
            self.full_buffer = True

    def non_terminals_idx(self, idx):
        nonzero_idx = torch.nonzero(self.non_terminals[idx]).view(-1)
        next_idx = self.order_next[idx[nonzero_idx]].clone()
        return next_idx, nonzero_idx

    def update(self, model, mcts, value_support = 3, reward_support = 1):
        print('buffer is updating')
        pred_values = torch.zeros((self.max_size, 1))
        count = self.max_size if self.full_buffer else self.index

        for idx in range(0, count, self.batch_size):
            last_idx = min(idx + self.batch_size, count)

            with torch.no_grad():
                _, root_policies, root_values = mcts.run(
                    self.states[idx:last_idx].to(self.device),
                    model,
                    training = False,
                    value_support = value_support,
                    reward_support = reward_support
                )

            self.policies[idx:last_idx] = root_policies.cpu()
            pred_values[idx:last_idx] = root_values.cpu()

        self.td_values[0:count] = self.td_rewards[0:count] \
            + self.td_non_terminals[0:count] * self.td_gamma \
            * pred_values[self.td_next[0:count]]

    def update_targets(self, model, start, end, value_support):
        pred_values = torch.zeros((self.max_size, 1))

        for idx in range(start, end, self.batch_size):
            last_idx = min(idx + self.batch_size, end)

            with torch.no_grad():
                _, _, values = model(self.states[idx:last_idx].to(self.device))
                values = support_to_scalar(values, value_support)
            pred_values[idx:last_idx] = values.cpu()

        self.td_values[start:end] = self.td_rewards[start:end] \
            + self.td_non_terminals[start:end] * self.td_gamma \
            * pred_values[self.td_next[start:end]]
