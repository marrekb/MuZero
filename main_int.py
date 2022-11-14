import os
import torch
from core.muzero_int import MuZero
from core.mcts.gpu_int import MCTS
from core.buffer.base_buffer import ExperienceReplay
from models.model_atari96_int import Model
from models.rnd_atari import RND_Model
from utils.atari_wrapper import create_env

if __name__ == '__main__':
    name = 'mz96_int'
    path = 'results/PongNoFrameskip-v4/' + name + '/'

    batch_size = 1024
    buffer_size = 100        #final size is computed as batch_size * buffer_size
    lr = 0.0005
    lr_rnd = 2.5e-4

    max_env_steps = 1e8
    count_of_envs = 20

    value_ext_support = 3
    value_int_support = 3
    reward_support = 1

    td_steps = 10

    if os.path.isdir(path):
        print('directory has already existed')
    else:
        os.mkdir(path)
        print('new directory has been created')

    env_params = dict(env_name = 'PongNoFrameskip-v4', y1 = 33, y2 = 195, x1 = 0,
                      x2 = 160, denominator = 236.0, skip = 4, penalty = False,
                      steps_after_reset = 15, steps_after_negative_reward = 15,
                      max_steps = 2000)

    env = create_env(**env_params)
    input_dimension = env.reset().shape
    dim1, dim2, dim3 = input_dimension
    count_of_actions = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    model = Model(count_of_actions = count_of_actions,
                  value_ext_support = value_ext_support,
                  value_int_support = value_int_support,
                  reward_support = reward_support)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    rnd_model = RND_Model(dim2 * dim3)
    rnd_model.to(device)
    rnd_optimizer = torch.optim.Adam(rnd_model.parameters(), lr = lr_rnd)


    mcts = MCTS(count_of_actions = count_of_actions, device = device, filters = 96)
    buffer = ExperienceReplay(buffer_size = buffer_size, batch_size = batch_size,
                              count_of_actions = count_of_actions, td_steps = td_steps,
                              input_dimension = input_dimension, device = device)

    muzero = MuZero(model = model, optimizer = optimizer, 
                    rnd_model = rnd_model, rnd_optimizer = rnd_optimizer,
                    buffer = buffer,
                    mcts = mcts, device = device, path = path, name = name,
                    value_ext_support = value_ext_support,
                    value_int_support = value_int_support,
                    reward_support = reward_support)

    muzero.train(env_params = env_params, env_func = create_env,
                 count_of_actions = count_of_actions,
                 count_of_envs = count_of_envs,
                 max_env_steps = max_env_steps,
                 input_dimension = input_dimension)
