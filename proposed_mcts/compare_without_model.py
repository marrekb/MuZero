import torch
import numpy as np
import datetime
from torch.multiprocessing import set_start_method
import mz_model
import methods.GPU as GPU
import methods.CPU_GPU_S as CPU_GPU_S
import methods.CPU_GPU_P as CPU_GPU_P


def write_to_file(name, data):
    with open(name + '.txt', 'w') as file:
        for d in data:
            file.write(str(d) + '\n')

def predict_from_observations_best_mode(observations, model):
    l = len(observations)
    policies = torch.zeros((l, 18), device = observations.device)
    policies[:, 0] = 1
    return torch.zeros((l, 256, 6, 6), device = observations.device), \
           policies, torch.zeros((l, 1), device = observations.device)

def predict_from_observations_train_mode(observations, model):
    l = len(observations)
    return torch.zeros((l, 256, 6, 6), device = observations.device), \
           torch.rand((l, 18), device = observations.device), \
           torch.rand((l, 1), device = observations.device)

def predict_from_states_best_mode(states, actions, model):
    l = len(states)
    policies = torch.zeros((l, 18), device = states.device)
    policies[:, 0] = 1
    return states, torch.zeros((l, 1), device = states.device), policies, \
           torch.zeros((l, 1), device = states.device)

def predict_from_states_train_mode(states, actions, model):
    l = len(states)
    return states, torch.zeros((l, 1), device = states.device), \
                   torch.rand((l, 18), device = states.device), \
                   torch.rand((l, 1), device = states.device)

if __name__ == '__main__':
    count_of_replications = 100

    envs = [50, 100, 250, 500, 750]     #C_T
    approaches = ['GPU', 'CPU_GPU_P2', 'CPU_GPU_P5', 'CPU_GPU_P10', 'CPU_GPU_S']

    count_of_actions = 18
    observation_dim = (128, 96, 96)

    features_dim = (6, 6)
    dirichlet_alpha = 0.25

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    set_start_method('spawn')

    mz_params = dict(count_of_actions = count_of_actions, filters = 256,
                     features_dim = features_dim, device = device,
                     count_of_simulations = 50, c1 = 1.25, c2 = 19652.0,
                     gamma = 0.997, dirichlet_alpha = 0.25, exploration_fraction = 0.25)

    mz_gpu = GPU.MuZero(**mz_params)
    mz_cpu_gpu_p = CPU_GPU_P.MuZero(**mz_params)
    mz_cpu_gpu_s = CPU_GPU_S.MuZero(**mz_params)

    model = None

    for count_of_envs in envs:
        observations = torch.ones(((count_of_envs, ) + observation_dim), device = device)
        print('count of envs', count_of_envs)

        print('train mode')

        data = []
        for i in range(count_of_replications):
            with torch.no_grad():
                start = datetime.datetime.now()
                mz_gpu.mcts(
                    observations,
                    model,
                    predict_from_observations_train_mode,
                    predict_from_states_train_mode,
                    training = True
                )
                data.append((datetime.datetime.now() - start).total_seconds())
        result = np.average(data)
        write_to_file('GPU_t_' + str(count_of_envs), data)
        print('GPU', result)

        for count_of_cpus in [2, 5, 10, 25]:
            data = []
            for i in range(count_of_replications):
                with torch.no_grad():
                    start = datetime.datetime.now()
                    mz_cpu_gpu_p.mcts(
                        observations,
                        model,
                        predict_from_observations_train_mode,
                        predict_from_states_train_mode,
                        training = True,
                        cpu = count_of_cpus
                    )
                    data.append((datetime.datetime.now() - start).total_seconds())
            result = np.average(data)
            write_to_file('CPU_GPU_P' + str(count_of_cpus) + '_t_' + str(count_of_envs), data)
            print('CPU_GPU_P', count_of_cpus, result)

        data = []
        for i in range(count_of_replications):
            with torch.no_grad():
                start = datetime.datetime.now()
                mz_cpu_gpu_s.mcts(
                    observations,
                    model,
                    predict_from_observations_train_mode,
                    predict_from_states_train_mode,
                    training = True
                )
                data.append((datetime.datetime.now() - start).total_seconds())
        result = np.average(data)
        write_to_file('CPU_GPU_S_t_' + str(count_of_envs), data)
        print('CPU_GPU_S  ', result)

        print('best mode')
        data = []
        for i in range(count_of_replications):
            with torch.no_grad():
                start = datetime.datetime.now()
                mz_gpu.mcts(
                    observations,
                    model,
                    predict_from_observations_best_mode,
                    predict_from_states_best_mode,
                    training = True
                )
                data.append((datetime.datetime.now() - start).total_seconds())
        result = np.average(data)
        write_to_file('GPU_b_' + str(count_of_envs), data)
        print('GPU', result)

        for count_of_cpus in [2, 5, 10, 25]:
            data = []
            for i in range(count_of_replications):
                with torch.no_grad():
                    start = datetime.datetime.now()
                    mz_cpu_gpu_p.mcts(
                        observations,
                        model,
                        predict_from_observations_best_mode,
                        predict_from_states_best_mode,
                        training = True,
                        cpu = count_of_cpus
                    )
                    data.append((datetime.datetime.now() - start).total_seconds())
            result = np.average(data)
            write_to_file('CPU_GPU_P' + str(count_of_cpus) + '_b_' + str(count_of_envs), data)
            print('CPU_GPU_P', count_of_cpus, result)

        data = []
        for i in range(count_of_replications):
            with torch.no_grad():
                start = datetime.datetime.now()
                mz_cpu_gpu_s.mcts(
                    observations,
                    model,
                    predict_from_observations_best_mode,
                    predict_from_states_best_mode,
                    training = True
                )
                data.append((datetime.datetime.now() - start).total_seconds())
        result = np.average(data)
        write_to_file('CPU_GPU_S_b_' + str(count_of_envs), data)
        print('CPU_GPU_S', result)
