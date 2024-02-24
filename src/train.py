from gymnasium.wrappers.time_limit import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

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
    import numpy
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

    def save(self, path='./agent_state.pt'):
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        torch.save(self.model.state_dict(), path)


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
        self.model.load_state_dict(torch.load('./agent_state.pt', map_location = device))

    #Honteusement volé au nb 4 sur les DQN
    import random

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

    def greedy_action(network, state):

    class dqn_agent:
        def __init__(self, config, model):
            device = "cuda" if next(model.parameters()).is_cuda else "cpu"
            self.gamma = config['gamma']
            self.batch_size = config['batch_size']
            self.nb_actions = config['nb_actions']
            self.memory = ReplayBuffer(config['buffer_size'], device)
            self.epsilon_max = config['epsilon_max']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_stop = config['epsilon_decay_period']
            self.epsilon_delay = config['epsilon_delay_decay']
            self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
            self.model = model 
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        
        def gradient_step(self):
            if len(self.memory) > self.batch_size:
                X, A, R, Y, D = self.memory.sample(self.batch_size)
                QYmax = self.model(Y).max(1)[0].detach()
                #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
                update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
                QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
                loss = self.criterion(QXA, update.unsqueeze(1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step() 
        
        def train(self, env, max_episode):
            episode_return = []
            episode = 0
            episode_cum_reward = 0
            state, _ = env.reset()
            epsilon = self.epsilon_max
            step = 0

            while episode < max_episode:
                # update epsilon
                if step > self.epsilon_delay:
                    epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)

                # select epsilon-greedy action
                if np.random.rand() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = greedy_action(self.model, state)

                # step
                next_state, reward, done, trunc, _ = env.step(action)
                self.memory.append(state, action, reward, next_state, done)
                episode_cum_reward += reward

                # train
                self.gradient_step()

                # next transition
                step += 1
                if done:
                    episode += 1
                    print("Episode ", '{:3d}'.format(episode), 
                          ", epsilon ", '{:6.2f}'.format(epsilon), 
                          ", batch size ", '{:5d}'.format(len(self.memory)), 
                          ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                          sep='')
                    state, _ = env.reset()
                    episode_return.append(episode_cum_reward)
                    episode_cum_reward = 0
                else:
                    state = next_state

            return episode_return
