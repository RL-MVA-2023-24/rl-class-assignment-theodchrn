import matplotlib.pyplot as plt
from gymnasium.wrappers.time_limit import TimeLimit
# todo : possibilié loader modèle + argparser
from env_hiv import HIVPatient
import torch
import random
import numpy as np
from tqdm import tqdm
import sys
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')

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

    def __init__(self):

        print("Instanciation de l'agent")

        self.config = {'nb_actions': env.action_space.n,
                       'action_size': 1,
                       'state_dim': env.observation_space.shape[0],
                       'hidden_layers': 5,
                       'nb_neurons': 512,
                       'learning_rate': 0.001,
                       'gamma': 0.95,  # choisi d'après Ernst et al., 2006
                       'buffer_size': 100000,  # en-dessous de 5 000 on dirait qu'il n'apprend pas
                       'epsilon_min': 0.01,
                       'epsilon_max': 1.,
                       'epsilon_decay_period': 10000,
                       'epsilon_delay_decay': 400,
                       'batch_size': 1024,
                       'gradient_steps': 2,
                       'update_target_strategy': 'ema',  # 'ema', # or
                       'update_target_freq': 100,
                       'update_target_tau': 0.005,
                       'criterion': torch.nn.SmoothL1Loss(),
                       'monitoring_nb_trials': 20,
                       'monitor_every': 3,
                       'save_every': 50,
                       'save_always': 25,
                       'double': True,
                       'update_mem_every': 20,          # how often to update the priorities
                       'use_priority': True,
                       'PER_Buffer_eps': 0.7,
                       'PER_Buffer_Alpha': 0.5,
                       'PER_Buffer_Beta': 0.5,
                       }

        self.config['time'] = datetime.now().strftime("%Y%m%d-%H%M%S")
        if len(sys.argv) == 1:
            self.agent_name = 'DQN_Agent'
            from DQN_Agent import Agent as agent
            self.config['agent_name'] = self.agent_name
            self.config['max_episode'] = 20
            self.config['nb_episodes'] = 20
            self.agent = agent(self.config)
            self.path = os.getcwd() + "/models/best"

        elif len(sys.argv) >= 2:
            import importlib.util
            agent_file = os.getcwd() + "/" + sys.argv[1]  # get full path
            self.agent_name = os.path.basename(os.path.normpath(
                sys.argv[1][:-3]))  # get only specific module name
            spec = importlib.util.spec_from_file_location(
                self.agent_name, agent_file)  # delete trailing slashes and get module name
            module = importlib.util.module_from_spec(spec)  # import module
            spec.loader.exec_module(module)  # load module in its own workspace
            agent = getattr(module, 'Agent')  # load the Agent class as agent

            self.config['max_episode'] = int(sys.argv[2])
            self.config['nb_episodes'] = int(sys.argv[2])
            self.config['agent_name'] = self.agent_name
            if len(sys.argv) == 4:
                self.config['epsilon_max'] = .3
                self.config['epsilon_min'] = 5e-3
                self.config['buffer_size'] = 10000

            print(f"{self.config['agent_name']=}")
            self.agent = agent(self.config)
            self.path = os.getcwd() + "/models/{}_{}".format(self.agent_name,
                                                             self.config['time'])

            if len(sys.argv) == 4:
                self.agent.load(sys.argv[-1][:-3])

        print(f'{device=}')

    # def act(self, observation: np.ndarray, use_random: bool = False) -> int:
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
        return self.agent.act(observation, use_random)
#            with torch.no_grad():
#                Q = self.agent(torch.Tensor(observation).unsqueeze(0).to(device))
#                return torch.argmax(Q).item()

    # def save(self, path: str) -> None:

    def save(self, path=None):
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        path = self.path if None else path
        print(f"Sauvegarde du modèle à : {path}")
        self.agent.save(path)
        # torch.save(self.agent.model.state_dict(), self.path)

    # def load(self) -> None:

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
        # print(f"Chargement du modèle {self.path}")
        self.agent.load(self.path)

#        self.agent.model.load_state_dict(torch.load(self.path, map_location = device))
#        self.agent.eval()


def fill_buffer(env, agent, buffer_size):
    state, _ = env.reset()
    progress_bar = tqdm(total=buffer_size, desc="Filling the replay buffer")
    for _ in range(buffer_size):
        action = agent.act(state, use_random=False)
        next_state, reward, done, trunc, _ = env.step(action)
        agent.memory.append(state, action, reward, next_state, done)
        if done or trunc:
            state, _ = env.reset()
        else:
            state = next_state
        progress_bar.update(1)
    progress_bar.close()


if __name__ == "__main__":
    # Set the seed
    seed = np.random.randint(42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set the device

    # Fill the buffer
    FinalAgent = ProjectAgent()
    if FinalAgent.agent_name != 'PER_Agent':
        fill_buffer(env, FinalAgent.agent, FinalAgent.config['buffer_size'])

    ep_length, disc_rewards, tot_rewards, V0 = FinalAgent.agent.train(
        env, FinalAgent.config['max_episode'])
    FinalAgent.save(FinalAgent.path)

    plt.figure()
    plt.plot(ep_length, label="training episode length")
    if FinalAgent.config['monitoring_nb_trials'] > 0:
        plt.plot(tot_rewards, label="MC eval of total reward")
    plt.legend()
    plt.savefig(FinalAgent.path + '-fig1.png')

    if FinalAgent.config['monitoring_nb_trials'] > 0:
        plt.figure()
        plt.plot(disc_rewards, label="MC eval of discounted reward")
        plt.plot(V0, label="average $max_a Q(s_0)$")
        plt.legend()
        plt.savefig(FinalAgent.path + '-fig2.png')

    print("Agent trained and saved!")
