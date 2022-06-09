import torch
import datetime
from torch.multiprocessing import set_start_method
import mz_model
import methods.GPU as GPU


def write_to_file(name, data):
    with open(name + '_memory_model.txt', 'w') as file:
        for d in data:
            file.write(str(d) + '\n')

def predict_from_observations_train_mode(observations, model):
    l = len(observations)
    states = model.presentation_function(observations)
    model.predict_function(states)
    return states, torch.rand((l, 18), device = observations.device), \
           torch.rand((l, 1), device = observations.device)

def predict_from_states_train_mode(states, actions, model):
    l = len(states)
    actions_t = torch.zeros(((l, 1, 6,6)), device = states.device)
    states = torch.cat((states, actions_t), dim = 1)
    states, rewards = model.dynamic_function(states)
    model.predict_function(states)
    return states, torch.zeros((l, 1), device = states.device), torch.rand((l, 18), \
           device = states.device), torch.rand((l, 1), device = states.device)

if __name__ == '__main__':
    count_of_replications = 2

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

    model = mz_model.Model(count_of_actions = count_of_actions, features_size = 9216)
    model.to(device)

    for count_of_envs in envs:
        observations = torch.ones(((count_of_envs, ) + observation_dim), device = device)
        print('count of envs', count_of_envs)

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
