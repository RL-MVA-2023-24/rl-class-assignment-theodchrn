import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import trange
from evaluate import evaluate_HIV
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class policyNetwork(nn.Module):
    def __init__(self, state_dim, n_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, n_action)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores,dim=1)

    def sample_action(self, x):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.sample().item()

    def log_prob(self, x, a):
        probabilities = self.forward(x)
        action_distribution = Categorical(probabilities)
        return action_distribution.log_prob(a)


class valueNetwork(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(dim=0)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Agent:
    def __init__(self, config):
        self.policy = policyNetwork(config['state_dim'], config['nb_actions'])
        self.value = valueNetwork(config['state_dim'])
        self.nb_actions = config['nb_actions']
        self.nb_observation = config['state_dim']
        self.device = "cuda" if next(self.policy.parameters()).is_cuda else "cpu"
        self.scalar_dtype = next(self.policy.parameters()).dtype
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.99
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = torch.optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()),lr=lr)
        self.nb_episodes = config['nb_episodes'] if 'nb_episodes' in config.keys() else 1
        self.entropy_coefficient = config['entropy_coefficient'] if 'entropy_coefficient' in config.keys() else 0.001

        # Mise en place monitoring par MC
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.monitor_every = config['monitor_every'] if 'monitor_every' in config.keys() else 10
        self.save_every = config['save_every'] if 'save_every' in config.keys() else 100
    
    def save(self, path):
        torch.save(self.policy.state_dict(), path + "_policy.pt")
        torch.save(self.value.state_dict(), path + "_value.pt")

    def load(self, path):
        self.policy.load_state_dict(torch.load(path + "_policy.pt",map_location=device))
        self.value.load_state_dict(torch.load(path + "_value.pt",map_location=device))
        self.policy.eval()


    def sample_action(self, x):
        probabilities = self.policy(torch.as_tensor(x,dtype=torch.float32))
        action_distribution = Categorical(probabilities)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()
        return action.item(), log_prob, entropy
    
    def one_gradient_step(self, env):
        # run trajectories until done
        episodes_sum_of_rewards = []
        log_probs = []
        returns = []
        values = []
        entropies = []
        for ep in range(self.nb_episodes):
            x,_ = env.reset()
            rewards = []
            episode_cum_reward = 0
            while(True):
                a, log_prob, entropy = self.sample_action(x)
                y,r,d,t,_ = env.step(a)
                values.append(self.value(torch.as_tensor(x,dtype=torch.float32)).squeeze(dim=0))
                log_probs.append(log_prob)
                entropies.append(entropy)
                rewards.append(r)
                episode_cum_reward += r
                x=y
                if d or t:
                    # compute returns-to-go
                    new_returns = []
                    G_t = self.value(torch.as_tensor(x,dtype=torch.float32)).squeeze(dim=0)
                    for r in reversed(rewards):
                        G_t = r + self.gamma * G_t
                        new_returns.append(G_t)
                    new_returns = list(reversed(new_returns))
                    returns.extend(new_returns)
                    episodes_sum_of_rewards.append(episode_cum_reward)
                    break
        # make loss        
        returns = torch.cat(returns)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        advantages = returns - values
        pg_loss = -(advantages.detach() * log_probs).mean()
        entropy_loss = -entropies.mean()
        critic_loss = advantages.pow(2).mean()
        loss = pg_loss + critic_loss + self.entropy_coefficient * entropy_loss
        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return np.mean(episodes_sum_of_rewards)

    def greedy_action(self, observation):
        with torch.no_grad():
            a = self.policy.sample_action(torch.as_tensor(observation, dtype=torch.float32))
        return a

    def act(self, observation, use_random=False):
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            return self.greedy_action(observation)


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
                val.append(self.value(torch.as_tensor(x, dtype=torch.float32)))
        return np.mean(val)

    def train(self, env, nb_rollouts):
        avg_sum_rewards = []
        MC_avg_total_reward = []   
        MC_avg_discounted_reward = []   
        V_init_state = []
        best_return = 0

        for ep in trange(nb_rollouts):
            avg_sum_rewards.append(self.one_gradient_step(env))
            validation_score = evaluate_HIV(agent=self, nb_episode=1)
            
            if self.monitoring_nb_trials>0: 
                MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)    
                V0 = self.V_initial_state(env, self.monitoring_nb_trials)   
                MC_avg_total_reward.append(MC_tr)   
                MC_avg_discounted_reward.append(MC_dr)   
                V_init_state.append(V0)   
                print("Episode ", '{:2d}'.format(ep), 
                      ", ep return ", '{:6.0f}'.format(avg_sum_rewards[-1]), 
                      ", Validation score ", '{:4.6g}'.format(validation_score),
                      ", MC tot ", '{:6.0f}'.format(MC_tr),
                      ", MC disc ", '{:6.0f}'.format(MC_dr),
                      ", V0 ", '{:6.0f}'.format(V0),
                      sep='')
                if MC_tr > best_return:
                    best_return = MC_tr
                    self.best_policy = deepcopy(self.policy).to(device)
                    self.best_value =  deepcopy(self.value).to(device)
                    torch.save(self.best_policy.state_dict(), "best_policy.pt")
                    torch.save(self.best_value.state_dict(), "best_value.pt")
                    print(f"Saving model! Current validation score : {validation_score:.6g}\n")
                    print("Saving model! Best return is updated to ", best_return)

        return avg_sum_rewards, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
