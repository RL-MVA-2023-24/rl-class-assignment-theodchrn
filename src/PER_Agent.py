import gymnasium as gym
from env_hiv import HIVPatient
from evaluate import evaluate_HIV_population
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
import os
from utils import device, set_seed
from buffer import ReplayBuffer, PrioritizedReplayBuffer


class MLP:
    def __init__(self, state_size, hidden_dim, action_size):
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_size)
        ).to(device())


class Agent:
    def __init__(self, config):
        self.time = config['time']
        self.name = config['agent_name']
        self.nb_actions = config['nb_actions']
        self.action_size = config['action_size']
        self.nb_observation = config['state_dim']
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']

        self.use_priority = config['use_priority'] if 'use_priority' in config.keys() else True
        self.buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else 10
        self.alpha = config['PER_Buffer_Alpha'] if 'PER_Buffer_Alpha' in config.keys() else 0.1
        self.beta = config['PER_Buffer_Beta'] if 'PER_Buffer_Beta' in config.keys() else 0.1
        self.eps = config['PER_Buffer_eps'] if 'PER_Buffer_eps' in config.keys() else 1e-3
        if self.use_priority:
            self.memory = PrioritizedReplayBuffer(self.nb_observation, self.action_size, self.buffer_size, self.eps, self.alpha, \
                    self.beta)
        else:
            self.memory = ReplayBuffer(self.nb_observation, self.nb_actions, self.buffer_size)

        self.nb_actions = config['nb_actions']
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        
        self.model = MLP(self.nb_observation, config['nb_neurons'], self.nb_actions).model

        ## mise en place d'un target network.
        self.target_model = deepcopy(self.model).to(device())
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001

        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer2 = config['optimizer2'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)

        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        # Mise en place monitoring par MC
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitor_every = config['monitor_every'] if 'monitor_every' in config.keys() else 10
        self.save_every = config['save_every'] if 'save_every' in config.keys() else 100
        self.double = config['double'] if 'double' in config.keys() else True
        self.save_always = config['save_always'] if 'save_always' in config.keys() else 20

    def soft_update(self):
            if self.update_target_strategy == 'ema':
                for tp, sp in zip(self.target_model.parameters(), self.model.parameters()):
                    tp.data.copy_((1 - self.tau) * tp.data + self.tau * sp.data)
            elif self.update_target_strategy == 'replace':
                self.target_model.load_state_dict(self.model.state_dict())



    def act(self, state, use_random = False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            with torch.no_grad():
                state = torch.as_tensor(state, dtype=torch.float).to(device())
                action = torch.argmax(self.model(state)).cpu().numpy().item()
        return action

    def update(self, batch, weights=None):
        state, action, reward, next_state, done = batch

        Q_next = self.target_model(next_state).max(dim=1).values
        Q_target = reward + self.gamma * (1 - done) * Q_next
        Q = self.model(state)[torch.arange(len(action)), action.to(torch.long).flatten()]

        assert Q.shape == Q_target.shape, f"{Q.shape}, {Q_target.shape}"

        if weights is None:
            weights = torch.ones_like(Q)

        td_error = torch.abs(Q - Q_target).detach()
        loss = torch.mean((Q - Q_target)**2 * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            self.soft_update()

        return loss.item(), td_error

    def save(self, path):
        torch.save(self.model.state_dict(), path + ".pt")

    def load(self, path):
        print(f"loading model {path}")
        self.model.load_state_dict(torch.load(path + ".pt",map_location=device()))
        self.target_model = deepcopy(self.model).to(device())

    def evaluate_policy(self, env, episodes=5, seed=0):
        set_seed(env, seed=seed)

        returns = []
        for ep in range(episodes):
            done, total_reward = False, 0
            state, _ = env.reset(seed=seed + ep)

            while not done:
                state, reward, terminated, truncated, _ = env.step(self.act(state))
                done = terminated or truncated
                total_reward += reward
            returns.append(total_reward)
        return np.mean(returns), np.std(returns)


    def train(self, env, max_episode = None): 

        max_episode = 20 if None else max_episode

        self.previous_raw = None


        if isinstance(self.memory, PrioritizedReplayBuffer):
            print(f"Prioritizing Experience Replay!")
        else:
            print("Warning not prioritizing!!")
        episode_return = []
        episode = 0
        step=0
        best_return = 0
        validation_score = 0

        batch_size=self.batch_size
        eps =self.epsilon_max 
        test_every=self.monitor_every
        seed=np.random.randint(42)
        print(f"Training on: {env}, Device: {device()}, Seed: {seed}")


        rewards_total, stds_total = [], []
        loss_count, total_loss = 0, 0

        best_reward = -np.inf

        done = False
        state, _ = env.reset(seed=seed)

        def generator():
            while episode < max_episode:
                yield

        for _ in tqdm(generator()):

            if step > self.epsilon_delay:
                eps = max(self.epsilon_min, eps-self.epsilon_step)

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = self.act(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            step += 1
            done = terminated or truncated
            self.memory.append(state, action, reward, next_state, int(done))

            state = next_state

            if step > batch_size:
                if isinstance(self.memory, ReplayBuffer):
                    batch = self.memory.sample(batch_size)
                    loss, td_error = self.update(batch)
                elif isinstance(self.memory, PrioritizedReplayBuffer):
                    batch, weights, tree_idxs = self.memory.sample(batch_size)
                    loss, td_error = self.update(batch, weights=weights)

                    self.memory.update_priorities(tree_idxs, td_error.numpy())
                else:
                    raise RuntimeError("Unknown self.memory")

                total_loss += loss
                loss_count += 1

                if done:
                    done = False
                    state, _ = env.reset(seed=seed)
                    episode += 1
                    if episode % test_every == 0:
                        mean, std = self.evaluate_policy(env, episodes=10, seed=seed)
                        validation_score = evaluate_HIV_population(agent=self, nb_episode=5)
                        episode_return.append(validation_score)

                        print(f"Episode: {episode}, Validation score: {validation_score:4.6g}, Reward mean: {mean:.2f}, Reward std: {std:.2f}, Loss: {total_loss / loss_count:.4f}, Eps: {eps:.3f}")

                        if mean > best_reward:
                            best_reward = mean
                        if validation_score > best_return:
                            if self.previous_raw is not None:
                                os.remove(self.previous_raw)


                            best_return = validation_score
                            self.best_model = deepcopy(self.model).to(device())
                            torch.save(self.model.state_dict(), "best_raw_{}_model_{}_val_len_{}_{}.pt".format(self.name, self.time, \
                                    int(str(validation_score)[:2]), np.floor(np.log10(np.abs(validation_score))).astype(int)))

                            print(f"Saving model! Current validation score : {validation_score:.6g}\n")
                            print("Saving model! Best return is updated to ", best_reward)
                            self.previous_raw = "best_raw_{}_model_{}_val_len_{}_{}.pt".format(self.name, self.time, \
                                    int(str(validation_score)[:2]), np.floor(np.log10(np.abs(validation_score))).astype(int))

                    if episode % self.save_always == 0:
                        self.saved_model = deepcopy(self.model).to(device())
                        if os.path.isdir('/kaggle/working'):
                            print("Kaggle saving")
                            torch.save(self.saved_model.state_dict(), "/kaggle/working/saved_model_kaggle_{}_{}.pt".format(self.name, self.time))
                            #torch.save(self.saved_model.state_dict(), "saved_model_kaggle_{}.pt".format(self.time))
                        else:
                            print("basic saving.")
                            torch.save(self.saved_model.state_dict(), "saved_model_{}.pt".format(self.time))


                        rewards_total.append(mean)
                        stds_total.append(std)

        return np.array(rewards_total), np.array(stds_total), total_loss, 0
