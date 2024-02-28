from gymnasium.wrappers.time_limit import TimeLimit
import time
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
from collections import deque
from copy import deepcopy
from datetime import datetime
import matplotlib.pyplot as plt
from copy import deepcopy
import math
from replay_buffer_prioritized import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 2, activation = torch.nn.SiLU(), normalization = None):
        super(MLP, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(depth - 1)])
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        if activation is not None:
            self.activation = activation
        else:
            self.activation = torch.nn.ReLU()
        if normalization == 'batch':
            self.normalization = torch.nn.BatchNorm1d(hidden_dim)
        elif normalization ==  'layer':
            self.normalization = torch.nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.normalization is not None:
                x = self.normalization(x)
        return self.output_layer(x)



class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, config, compute_weights = False):

        self.model = MLP(config['state_dim'], config['nb_neurons'], config['nb_actions'], depth = config['hidden_layers'], activation =nn.ReLU(), normalization = 'None').to(device)
        self.qnetwork_target = MLP(config['state_dim'], config['nb_neurons'], config['nb_actions'], depth = config['hidden_layers'], activation =nn.ReLU(), normalization = 'None').to(device)
        self.compute_weights = compute_weights
        self.time = config['time']
        self.buffer_size = config['buffer_size']
        self.name = config['agent_name']
        self.nb_actions = config['nb_actions']
        self.nb_observation = config['state_dim']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        ## mise en place d'un target network.
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001

        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = config['optimizer2'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)

        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        # Mise en place monitoring par MC
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitor_every = config['monitor_every'] if 'monitor_every' in config.keys() else 10
        self.save_every = config['save_every'] if 'save_every' in config.keys() else 100
        self.double = config['double'] if 'double' in config.keys() else True
        self.save_always = config['save_always'] if 'save_always' in config.keys() else 20
        self.update_mem_every = config['update_mem_every']
        self.update_mem_par_every = config['update_mem_par_every']
        self.experiences_per_sampling = math.ceil(self.batch_size * self.update_mem_every / self.monitor_every)

        # Q-Network
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Replay memory
        self.memory = ReplayBuffer(
            self.nb_actions, self.buffer_size, self.batch_size, self.experiences_per_sampling, seed=42, compute_weights=compute_weights)
        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.append(state, action, reward, next_state, done)
        
        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % self.monitor_every
        self.t_step_mem = (self.t_step_mem + 1) % self.update_mem_every
        self.t_step_mem_par = (self.t_step_mem_par + 1) % self.update_mem_par_every
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > self.experiences_per_sampling:
                sampling = self.memory.sample()
                self.learn(sampling, self.gamma)
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()


    def greedy_action(self, observation):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.model.eval()
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            return self.greedy_action(observation)

    def learn(self, sampling, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices  = sampling

        ## TODO: compute and minimize the loss        
        q_target = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        expected_values = rewards + gamma*q_target*(1-dones)
        output = self.model(states).gather(1, actions)
        loss = F.mse_loss(output, expected_values)
        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.model, self.qnetwork_target, self.update_target_tau)

        # ------------------- update priorities ------------------- #
        delta = abs(expected_values - output.detach()).numpy()
        #print("delta", delta)      
        self.memory.update_priorities(delta, indices)  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def train(self, env, n_episodes=None, max_t=200, eps_start=1.0, eps_end=0.001, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []                        # list containing scores from each episode
        validation_scores = []
        best = []
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        i_episode = 0
        n_episodes = 20 if None else n_episodes

        def generator():
            while i_episode < n_episodes:
                yield

        start_time = time.time()

        for _ in tqdm(generator()):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                next_state, reward, done, trunc, _ = env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done or trunc:
                    break 
            i_episode += 1
            current_val = evaluate_HIV(agent=self, nb_episode=1)
            validation_scores.append(current_val)
            if current_val > best:
                torch.save(self.model.state_dict(), "best_raw_model_{}_val_len_{}_{}.pt".format(self.time, int(str(current_val)[:2]), np.floor(np.log10(np.abs(current_val))).astype(int)))
                print(f'Saving raw model! Best val score is updated to {current_val}')
                best = current_val

            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                elapsed_time = time.time() - start_time
                print("Duration: ", elapsed_time)
            if np.mean(scores_window)>=1e10:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(self.model.state_dict(), 'checkpoint.pth')
                break
        elapsed_time = time.time() - start_time
        print("Training duration: ", elapsed_time)
        return scores, 0, 0, validation_scores



    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")

    def load(self, path):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        print(f"loading model {path}")
        self.model.load_state_dict(torch.load(path + ".pt",map_location=device))
        self.model.eval()




