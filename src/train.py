from gymnasium.wrappers.time_limit import TimeLimit
from env_hiv import HIVPatient
from evaluate import evaluate_HIV
import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    """
    Defines an interface for agents in a simulation or decision-making environment.

    An Agent must implement methods to act based on observations, save its state to a file,
    and load its state from a file. This interface uses the Protocol class from the typing
    module to specify methods that concrete classes must implement.

    Protocols are a way to define formal Python interfaces. They allow for type checking
    and ensure that implementing classes provide specific methods with the expected signatures.
    """
    #def act(self, observation: np.ndarray, use_random: bool = False) -> int:
    def act(self, observation, use_random=False):
        """
        Determines the next action based on the current observation from the environment.

        Implementing this method requires processing the observation and optionally incorporating
        randomness into the decision-making process (e.g., for exploration in reinforcement learning).

        Args:
            observation (np.ndarray): The current environmental observation that the agent must use
                                       to decide its next action. This array typically represents
                                       the current state of the environment.
            use_random (bool, optional): A flag to indicate whether the agent should make a random
                                         decision. This is often used for exploration. Defaults to False.

        Returns:
            int: The action to be taken by the agent.
        """
        if use_random:
            return np.random.randint(self.nb_actions)
        else:
            with torch.no_grad():
                Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
                return torch.argmax(Q).item()


    #def save(self, path: str) -> None:
    def save(self, path='agent_state.pt'):
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        torch.save(self.model.state_dict(), path)


    #def load(self) -> None:
    def load(self):
        """
        Loads the agent's state from a file specified by the path (HARDCODED). This not a good practice,
        but it will simplify the grading process.

        This method should deserialize the saved state (e.g., model weights, configuration settings)
        from the file and restore the agent to this state. Implementations must ensure that the
        agent's state is compatible with the `act` method's expectations.

        Note:
            It's important to ensure that neural network models (if used) are loaded in a way that is
            compatible with the execution device (e.g., CPU, GPU). This may require specific handling
            depending on the libraries used for model implementation. WARNING: THE GITHUB CLASSROOM
        HANDLES ONLY CPU EXECUTION. IF YOU USE A NEURAL NETWORK MODEL, MAKE SURE TO LOAD IT IN A WAY THAT
        DOES NOT REQUIRE A GPU.
        """
        self.model.load_state_dict(torch.load('agent_state.pt', map_location = device))





    def __init__(self, target = True):

        self.config = {'nb_actions': env.action_space.n ,
                       'state_dim': env.observation_space.shape[0],
                       'learning_rate': 0.001,
                       'gamma': 0.98, #choisi d'après Ernst et al., 2006
                       'buffer_size': 100000,
                       'epsilon_min': 0.01,
                       'epsilon_max': 1.,
                       'epsilon_decay_period': 17000,
                       'epsilon_delay_decay': 500,
                       'batch_size': 500,
                       'max_episode' : 200,
                       'gradient_steps': 1,
                       'update_target_strategy': 'replace', # or 'ema'
                       'update_target_freq': 400,
                       'update_target_tau': 0.005,
                       'criterion': torch.nn.SmoothL1Loss()}



        self.gamma = self.config['gamma']
        self.batch_size = self.config['batch_size']
        self.nb_actions = self.config['nb_actions']
        self.memory = ReplayBuffer(self.config['buffer_size'], device)
        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = self.DQN()

        ## Si mise en place d'un target network.
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = self.config['criterion'] if 'criterion' in self.config.keys() else torch.nn.MSELoss()
        lr = self.config['learning_rate'] if 'learning_rate' in self.config.keys() else 0.001
        self.optimizer = self.config['optimizer'] if 'optimizer' in self.config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = self.config['gradient_steps'] if 'gradient_steps' in self.config.keys() else 1
        self.update_target_strategy = self.config['update_target_strategy'] if 'update_target_strategy' in self.config.keys() else 'replace'
        self.update_target_freq = self.config['update_target_freq'] if 'update_target_freq' in self.config.keys() else 20
        self.update_target_tau = self.config['update_target_tau'] if 'update_target_tau' in self.config.keys() else 0.005
        print(f'{device=}')

        if target:
            self.gradient_step = self.gradient_step_target
        else:
            self.gradient_step = self.gradient_step_vanilla
        self.target = target

    def DQN(self):
        nb_neurons=300 

        # Ancienne version, bof
##        DQN = torch.nn.Sequential(nn.Linear(env.observation_space.shape[0], 120),
##                                  nn.ReLU(),
##                                  nn.Linear(120, 120),
##                                  nn.ReLU(),
##                                  nn.Linear(120, 84),
##                                  nn.ReLU(), 
##                                  nn.Linear(84, env.action_space.n)).to(device)
##
        DQN = torch.nn.Sequential(
            nn.Linear(self.config['state_dim'], nb_neurons),
           # nn.BatchNorm1d(nb_neurons),  # Batch Normalization layer after the input layer
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, nb_neurons),
           # nn.BatchNorm1d(nb_neurons),  # Batch Normalization layer after the first hidden layer
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, nb_neurons),
           # nn.BatchNorm1d(nb_neurons),  # Batch Normalization layer after the second hidden layer
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, nb_neurons),
           # nn.BatchNorm1d(nb_neurons),  # Batch Normalization layer after the third hidden layer
            nn.LeakyReLU(),
            nn.Linear(nb_neurons, self.config['nb_actions'])
            ).to(device)

        return DQN
    
    def gradient_step_vanilla(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 
    
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

    def train(self):
        max_episode = self.config['max_episode']
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        validation_base = 0

        for episode in tqdm(range(max_episode)):
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
            if self.target:
                if self.update_target_strategy == 'replace':
                    if step % self.update_target_freq == 0: 
                        self.target_model.load_state_dict(self.model.state_dict())
                if self.update_target_strategy == 'ema':
                    target_state_dict = self.target_model.state_dict()
                    model_state_dict = self.model.state_dict()
                    tau = self.update_target_tau
                    for key in model_state_dict:
                        target_state_dict[key] = tau*model_state_dict[key] + (1-tau)*target_state_dict[key]
                    target_model.load_state_dict(target_state_dict)

            # next transition
            step += 1
            if done or trunc:
                #episode += 1
                validation_score = evaluate_HIV(agent=self, nb_episode=1)
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", Validation score ", '{:4.1f}'.format(validation_score),
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
                if validation_score > validation_base:
                    validation_base = validation_base
                    self.best_model = deepcopy(self.model).to(device)
                    torch.save(self.best_model.state_dict(), "best_model.pt")

            else:
                state = next_state

        return episode_return


#Honteusement volé au nb 4 sur les DQN
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)


if __name__=='__main__':
    agent = ProjectAgent()
    agent.train()
    agent.save()

