from gymnasium.wrappers.time_limit import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import os
from copy import deepcopy
from buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=200)


class NoisyLinear(nn.Module):
    # Noisy Linear Layer for factorised Gaussian Noise, for better computational efficiency
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        # make the sigmas trainable:
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        # not trainable tensor for the nn.Module
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))


        # extra parameter for the bias and register buffer for the bias parameter
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
    
        # reset parameter as initialization of the layer
        self.reset_parameter()
        self.reset_noise()
    
    def reset_parameter(self):
        """
        On initialise les paramètres
        """
        std = math.sqrt(1/self.in_features)
        self.weight_mu.data.uniform_(-std, std)
        self.bias_mu.data.uniform_(-std, std)

        self.weight_sigma.data.fill_(self.sigma_init * std)
        self.bias_sigma.data.fill_(self.sigma_init * std)

    def reset_noise(self):
        epsilon_i = self.scale_noise(self.in_features)
        epsilon_j = self.scale_noise(self.out_features)
        self.weight_epsilon.copy_(torch.outer(epsilon_j, epsilon_i))
        self.bias_epsilon.copy_(epsilon_j)

    def scale_noise(self, size):
        x = torch.randn(size) # generate 0-mean noise with fixed statistic (Gaussian(O,1))
        x = x.sign().mul(torch.sqrt(x.abs())) # activation function
        return x
    
    def forward(self, input):
        #sample factorised gaussian noise
        self.reset_noise()

        weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
        bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)

        return F.linear(input, weight, bias)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth = 2, activation = nn.SiLU(), normalization = None):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, \
                hidden_dim) for _ in range(depth - 1)])
        if activation is not None:
            self.activation = activation
        else:
            self.activation = nn.ReLU()
        if normalization == 'batch':
            self.normalization = nn.BatchNorm1d(hidden_dim)
        elif normalization ==  'layer':
            self.normalization = nn.LayerNorm(hidden_dim)
        else:
            self.normalization = None

        self.nn_1 = NoisyLinear(hidden_dim, hidden_dim)
        self.nn_2 = NoisyLinear(hidden_dim, output_dim)


    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if self.normalization is not None:
                x = self.normalization(x)
        x = F.relu(self.nn_1(x))
        return self.nn_2(x)


class Agent:
    def __init__(self, config):
        self.model = MLP(input_dim=config['state_dim'], hidden_dim=config['nb_neurons'], \
                output_dim=config['nb_actions'], depth = config['hidden_layers'], \
                activation =nn.ReLU(), normalization = 'None').to(device)
        self.time = config['time']
        self.name = config['agent_name']
        self.nb_actions = config['nb_actions']
        self.nb_observation = config['state_dim']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        ## mise en place d'un target network.
        self.target_model = deepcopy(self.model).to(device)
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


    def greedy_action(self, observation):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            return self.greedy_action(observation)

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")

    def load(self, path):
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        print(f"loading model {path}")
        self.model.load_state_dict(torch.load(path + ".pt",map_location=device))
        self.target_model = deepcopy(self.model).to(device)

    def gradient_step_target(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def gradient_step_double(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            Q_target_Ymax = self.target_model(Y).max(1)[0].detach()
            Q_Ymax = self.model(Y).max(1)[0].detach()
            next_Q = torch.min(Q_target_Ymax, Q_Ymax)
            update = torch.addcmul(R, 1-D, next_Q, value=self.gamma)
            Q_target_XA = self.target_model(X).gather(1, A.to(torch.long).unsqueeze(1))
            Q_XA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            
            loss = self.criterion(Q_target_XA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
            
            loss = self.criterion(Q_XA, update.unsqueeze(1))
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step() 


    def MC_eval(self, env, nb_trials):
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x,_ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = self.greedy_action(x)
                y,r,done,trunc,_ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma**step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)


    def V_initial_state(self, env, nb_trials):
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x,_ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(device)).max().item())
        return np.mean(val)


    def gradient_step(self):
        if self.double:
            self.gradient_step_double()
        else:
            self.gradient_step_target()

    def train(self, env, max_episode = None):
        self.model.train()
        max_episode = 20 if None else max_episode
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        best_return = 0
        self.previous_raw = None

        MC_avg_total_reward = []   
        MC_avg_discounted_reward = []   
        V_init_state = []   


        def generator():
            while episode < max_episode:
                yield

        for _ in tqdm(generator()):
            #while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = self.act(state, use_random=True)
            else:
                action = self.act(state, use_random=False)

            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()


            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0: 
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                self.target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                episode += 1
                validation_score = evaluate_HIV(agent=self, nb_episode=1)

                # Monitoring
                if episode % self.save_always == 0:
                    self.saved_model = deepcopy(self.model).to(device)
                    if os.path.isdir('/kaggle/working'):
                        print("Kaggle saving")
                        torch.save(self.saved_model.state_dict(), "/kaggle/working/saved_model_kaggle_{}_{}.pt".format(self.name, self.time))
                        #torch.save(self.saved_model.state_dict(), "saved_model_kaggle_{}.pt".format(self.time))
                    else:
                        print("basic saving.")
                        torch.save(self.saved_model.state_dict(), "saved_model_{}_{}.pt".format(self.name, self.time))


                if validation_score > best_return:
                    if self.previous_raw is not None:
                        os.remove(self.previous_raw)

                    best_return = validation_score
                    torch.save(self.model.state_dict(), "best_raw_{}_model_{}_val_len_{}_{}.pt".format(self.name, self.time, \
                            int(str(validation_score)[:2]), np.floor(np.log10(np.abs(validation_score))).astype(int)))
                    print("Saving raw model! Best return is updated to ", best_return)
                    self.previous_raw = "best_raw_{}_model_{}_val_len_{}_{}.pt".format(self.name, self.time, int(str(validation_score)[:2]), np.floor(np.log10(np.abs(validation_score))).astype(int))


                if self.monitoring_nb_trials>0 and episode % self.monitor_every == 0: 
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)   
                    MC_avg_total_reward.append(MC_tr)   
                    MC_avg_discounted_reward.append(MC_dr)   
                    V_init_state.append(V0)   
                    episode_return.append(episode_cum_reward)   
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.4f}'.format(epsilon), 
                          ", memory size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:6.0f}'.format(episode_cum_reward), 
                          ", Validation score ", '{:4.6g}'.format(validation_score),
                          ", MC tot ", '{:6.0f}'.format(MC_tr),
                          ", MC disc ", '{:6.0f}'.format(MC_dr),
                          ", V0 ", '{:6.0f}'.format(V0),
                          sep='')
                    if validation_score > best_return:
                        best_return = validation_score
                        self.best_model = deepcopy(self.model).to(device)
                        torch.save(self.best_model.state_dict(), "best_{}_{}_val_len_{}.pt".format(\
                                self.name, self.time, \
                                np.floor(np.log10(np.abs(validation_score))).astype(int)))
                        print(f"Saving model! Current validation score : {validation_score:.6g}\n")
                        print("Saving model! Best return is updated to ", best_return)
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode), 
                          ", epsilon ", '{:6.4f}'.format(epsilon), 
                          ", memory size ", '{:4d}'.format(len(self.memory)), 
                          ", ep return ", '{:6.0f}'.format(episode_cum_reward), 
                          ", Validation score ", '{:4.6g}'.format(validation_score),
                          sep='')

                # Start training again
                state, _ = env.reset()
                episode_cum_reward = 0

            else:
                state = next_state

        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
