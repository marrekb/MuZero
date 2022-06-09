import gym
import numpy as np
import cv2

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, skip, penalty, steps_after_reset,
                 steps_after_negative_reward, max_steps):
        super(EnvWrapper, self).__init__(env)
        self.skip = skip
        self.lives = 0
        self.penalty = penalty
        self.steps_after_reset = steps_after_reset * skip
        self.steps_after_negative_reward = steps_after_negative_reward * skip
        self.current_step = 0 #current step without reward
        self.max_steps = max_steps

    def step(self, action):
        sum_of_rewards = 0.0
        terminal = False

        for _ in range(self.skip):
            observation, reward, terminal, info = self.env.step(action)
            sum_of_rewards += reward
            if self.penalty:
                sum_of_rewards += self.ale.lives() - self.lives
                self.lives = self.ale.lives()
            if terminal:
                break

        if not terminal and sum_of_rewards < 0:
            for _ in range(self.steps_after_negative_reward):
                observation, _, terminal, info = self.env.step(self.env.action_space.sample())
        
        if int(sum_of_rewards) == 0:
            self.current_step += 1
            if self.current_step == self.max_steps:
                terminal = True
        else:
            self.current_step = 0

        return observation, sum_of_rewards, terminal, info

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.steps_after_reset):
            observation, _, _, _ = self.env.step(self.env.action_space.sample())
        self.lives = self.ale.lives()
        self.current_step = 0
        return observation

class PreProcessWrapper(gym.ObservationWrapper):
    def __init__(self, env, y1, y2, x1, x2, denominator):
        super(PreProcessWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0,
            shape = (4, 96, 96), dtype=np.float32)

        self.y1 = y1
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
        self.denominator = denominator

    def reset(self):
        self.buffer = np.zeros((4, 96, 96))
        return self.observation(self.env.reset())

    def observation(self, observation):
        # Don't use PIL for image processing, it takes long time...
        observation = cv2.cvtColor(observation[self.y1:self.y2, self.x1:self.x2], cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, (96, 96), interpolation=cv2.INTER_AREA)
        observation = observation.reshape(1, 96, 96)

        self.buffer[1:] = self.buffer[:3]
        self.buffer[0] = observation / self.denominator
        return self.buffer

def create_env(env_name = 'Pong-v0', y1 = 0, y2 = 210, x1 = 0, x2 = 160,
               penalty = False, denominator = 255.0, skip = 4,
               steps_after_reset = 0, steps_after_negative_reward = 0,
               max_steps = 1e7):
    env = gym.make(env_name)
    env = EnvWrapper(env, skip, penalty, steps_after_reset,
        steps_after_negative_reward, max_steps)
    env = PreProcessWrapper(env, y1, y2, x1, x2, denominator)
    return env
