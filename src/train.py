from gymnasium.wrappers.time_limit import TimeLimit
from env_hiv import HIVPatient
import torch
import random
import numpy as np
from tqdm import tqdm
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt

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

        self.config = {'nb_actions': env.action_space.n ,
                       'state_dim': env.observation_space.shape[0],
                       'hidden_layers' : 6,
                       'nb_neurons' : 512,
                       'learning_rate': 0.001,
                       'gamma': 0.98, #choisi d'après Ernst et al., 2006
                       'buffer_size': 1000,
                       'epsilon_min': 0.01,
                       'epsilon_max': 1.,
                       'epsilon_decay_period': 17000,
                       'epsilon_delay_decay': 500,
                       'batch_size': 500,
                       'gradient_steps': 1,
                       'update_target_strategy': 'replace', # or 'ema'
                       'update_target_freq': 400,
                       'update_target_tau': 0.005,
                       'criterion': torch.nn.SmoothL1Loss(),
                       'monitoring_nb_trials': 50, 
                       'monitor_every': 50, 
                       'save_path': './dqn_agent.pth',
                       'save_every': 50
                       }

        if len(sys.argv) == 3:
            self.config['max_episode'] = int(sys.argv[3])
        else:
            self.config['max_episode'] = 20


        time = datetime.now().strftime("%Y%m%d-%H%M%S")
        if len(sys.argv) == 1:
            from DQN_Agent import Agent as agent
            self.agent = agent(self.config)
            self.path = os.getcwd() + "/dqn_agent_{}.pt".format(time)

        elif len(sys.argv) >= 2:
            import importlib.util
            agent_file = os.getcwd() + "/" + sys.argv[1] # get full path
            spec = importlib.util.spec_from_file_location(os.path.basename(os.path.normpath(sys.argv[1][:-3])), agent_file) #delete trailing slashes and get module name
            module = importlib.util.module_from_spec(spec) # import module
            spec.loader.exec_module(module) #load module in its own workspace
            agent = getattr(module, 'Agent') #load the Agent class as agent

            self.agent = agent(self.config)
            self.path = os.getcwd() + "/{}_{}.pt".format(sys.argv[1], time)

        print(f'{device=}')

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
        return self.agent.act(observation, use_random)
#            with torch.no_grad():
#                Q = self.agent(torch.Tensor(observation).unsqueeze(0).to(device))
#                return torch.argmax(Q).item()


    #def save(self, path: str) -> None:
    def save(self, path):
        """
        Saves the agent's current state to a file specified by the path.

        This method should serialize the agent's state (e.g., model weights, configuration settings)
        and save it to a file, allowing the agent to be later restored to this state using the `load` method.

        Args:
            path (str): The file path where the agent's state should be saved.

        """
        print(f"Sauvegarde du modèle à : {self.path}")
        torch.save(self.agent.state_dict(), self.path)


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
        print(f"Chargement du modèle {self.path}")
        self.agent.load_state_dict(torch.load(self.path, map_location = device))
        self.agent.eval()


def fill_buffer(env, agent, buffer_size):
    state, _ = env.reset()
    progress_bar = tqdm(total=buffer_size, desc="Filling the replay buffer")
    for _ in range(buffer_size):
        action = agent.act(state)
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
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #env.seed(seed)
    # Set the device

    # Fill the buffer
    FinalAgent = ProjectAgent()
    fill_buffer(env, FinalAgent.agent,FinalAgent.config['buffer_size']) 

    ep_length, disc_rewards, tot_rewards, V0 = FinalAgent.agent.train(env, FinalAgent.config['max_episode'])
    FinalAgent.save(FinalAgent.path)


    plt.figure()
    plt.plot(ep_length, label="training episode length")
    if self.config['monitoring_nb_trials']>0:
        plt.plot(tot_rewards, label="MC eval of total reward")
    plt.legend()
    plt.savefig(self.path+'-fig1.png')

    if self.config['monitoring_nb_trials']>0:
        plt.figure()
        plt.plot(disc_rewards, label="MC eval of discounted reward")
        plt.plot(V0, label="average $max_a Q(s_0)$")
        plt.legend()
        plt.savefig(self.path+'-fig2.png')

    print("Agent trained and saved!")
