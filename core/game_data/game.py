import torch 

class Game:
    def __init__(self, env, seed):
        self.env = env 
        self.env.seed(seed)

        self.states = []
        self.policies = []
        self.actions = []
        self.rewards = []
        self.non_terminals = []

        self.score = 0

    def reset(self):
        state = torch.from_numpy(self.env.reset()).float()

        self.states.clear()
        self.policies.clear()
        self.actions.clear()
        self.rewards.clear()
        self.non_terminals.clear()

        self.score = 0
        return state

    def step(self, action, policy):
        self.policies.append(policy)
        self.actions.append(action)

        obs, reward, terminal, _ = self.env.step(action)
        obs = torch.from_numpy(obs).float()

        self.rewards.append(max(min(reward, 1.0), -1.0))
        self.score += reward 

        if terminal:
            self.non_terminals.append(0)
        else:
            if reward < 0:
                self.non_terminals.append(0)
            else:
                self.non_terminals.append(1)
        return obs, terminal 
    
    def compute_td(self, td_steps, gamma):
        length = len(self.rewards)
        td_rewards = torch.zeros(length)
        td_non_terminals = torch.ones(length, dtype = torch.int)

        for i in range(length):
            td_r = 0
            for j in range(0, td_steps):
                step = i + j
                td_r += gamma**j * self.rewards[step]

                if self.non_terminals[step] == 0:
                    td_non_terminals[i] = 0
                    break

            if td_non_terminals[i] == 1:
                last_step = i + td_steps

            td_rewards[i] = td_r

        return td_rewards.view(-1, 1), td_non_terminals.view(-1, 1)


