import math
import numpy as np

def write_to_file(log, filename):
    f = open(filename, "w")
    f.write(log)
    f.close()

class MovingAverageScore:
    def __init__(self, count = 100):
        self.memory = np.zeros(count)

        self.index = 0
        self.memory_size = count
        self.full_memory = False

        self.best_avg_score = - math.inf
        self.best_score = - math.inf
        self.count_of_episodes = 0

    def add(self, scores):
        length = len(scores)
        if length > 0:
            scores = np.array(scores)
            self.best_score = max(self.best_score, scores.max())
            self.count_of_episodes += length

            if length + self.index <= self.memory_size:
                new_index = self.index + length
                self.memory[self.index:new_index] = scores

                if new_index == self.memory_size:
                    self.index = 0
                    self.full_memory = True
                else:
                    self.index = new_index
            else:
                length_to_end = self.memory_size - self.index
                length_from_start = length - length_to_end

                self.memory[self.index:] = scores[:length_to_end]
                self.memory[:length_from_start] = scores[length_to_end:]

                self.index = length_from_start
                self.full_memory = True

    def mean(self):
        if self.full_memory:
            mean = self.memory.mean()
        else:
            if self.index == 0:
                return -math.inf, False
            mean = self.memory[:self.index].mean()

        if self.best_avg_score < mean:
            self.best_avg_score = mean
            return mean, True
        return mean, False

    def get_best_avg_score(self):
        return self.best_avg_score

    def get_count_of_episodes(self):
        return self.count_of_episodes

    def get_best_score(self):
        return self.best_score
