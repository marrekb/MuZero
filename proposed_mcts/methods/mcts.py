import torch
import math

class Node:
    def __init__(self, state, probs, device):
        self.S = state
        dim0 = probs.shape
        self.P = probs
        self.Q = torch.zeros(dim0, device = device)
        self.R = torch.zeros(dim0, device = device)
        self.N = torch.zeros(dim0, device = device, dtype = torch.int32)
        self.C = {}

    def getProbs(self):
        return self.N.float() / self.N.sum()

    def getRootData(self):
        probs = self.N.float() / self.N.sum()
        value = probs * self.Q
        return probs, value.sum()

class MCTS:
    def __init__(self, root, gamma, c1 = 1.25, c2 = 19652.0):
        self.c1 = c1
        self.c2 = c2
        self.gamma = gamma

        self.root = root
        self.parents = []

    def selection(self):
        parents = []
        node = self.root

        while True:
            N_sum = node.N.sum().item()
            sq = max(math.sqrt(float(N_sum)), 1.0)
            c = self.c1 + math.log((self.c2 + N_sum + 1) / self.c2)
            q_min, q_max = node.Q.min().item(), node.Q.max().item()
            norm_q = (node.Q - q_min) / (q_max - q_min + 0.0001)
            u = norm_q + node.P * (sq / (1.0 + node.N)) * c
            index = torch.argmax(u).item()

            parents.append((node, index))

            if index in node.C:
                node = node.C[index]
            else:
                parents.reverse()
                self.parents = parents
                return (node.S, index)

    def backup(self, v):
        for parent, i in self.parents:
            v = parent.R[i] + self.gamma * v
            count = parent.N[i] + 1
            parent.Q[i] = (parent.N[i] * parent.Q[i] + v) / count
            parent.N[i] = count
