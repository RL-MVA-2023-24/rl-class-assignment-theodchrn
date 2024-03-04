import numpy as np
from tqdm import tqdm
import pickle
from sklearn.ensemble import RandomForestRegressor

from best_model_david import greedy_action

"""
Fitted Q-Iteration (FQI) algorithm.

Approach inspired by "Clinical data based optimal STI strategies for HIV: a reinforcement learning approach", Ernst et al., 2006.
This implementation uses RandomForests instead of ExtraTrees.
"""


class Agent:
    def __init__(self, config):
        pass

    def file_extension(self):
        """
        Return file extension with which models are saved.
        """
        return ".pkl"

    def name(self):
        """
        Return name of the algorithm.
        """
        return "FQI"

    def save(self, model, path):
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def load(self, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def train(self, env, nb_actions, gamma=0.98):
        # The first step is to collect and store a dataset of samples
        self.S, self.A, self.R, self.S2, self.D = self.collect_samples(
            env, int(1e4))
        print("nb of collected samples:", self.S.shape[0])

        # Build the sequence of AVI Q-functions, learned using random forests
        nb_iter = 10
        Qfunctions = self.rf_fqi(
            nb_iter, nb_actions, gamma, disable_tqdm=False)
        self.Q = Qfunctions[-1]

        return Qfunctions[-1]

    def collect_samples(
            self,
            env,
            horizon,
            disable_tqdm=False,
            print_done_states=False):
        """
        The first step to perform FQI is to collect and store a dataset of samples.
        """
        s, _ = env.reset()
        # dataset = []
        S = []
        A = []
        R = []
        S2 = []
        D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = env.action_space.sample()
            s2, r, done, trunc, _ = env.step(a)
            # dataset.append((s,a,r,s2,done,trunc))
            S.append(s)
            A.append(a)
            R.append(r)
            S2.append(s2)
            D.append(done)
            if done or trunc:
                s, _ = env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        S = np.array(S)
        A = np.array(A).reshape((-1, 1))
        R = np.array(R)
        S2 = np.array(S2)
        D = np.array(D)
        return S, A, R, S2, D

    def rf_fqi(self, iterations, nb_actions, gamma, disable_tqdm=False):
        """
        Build the sequence of AVI Q-functions, learned using random forests.
        """
        nb_samples = self.S.shape[0]
        Qfunctions = []
        SA = np.append(self.S, self.A, axis=1)
        for iter in tqdm(range(iterations), disable=disable_tqdm):
            if iter == 0:
                value = self.R.copy()
            else:
                Q2 = np.zeros((nb_samples, nb_actions))
                for a2 in range(nb_actions):
                    A2 = a2 * np.ones((self.S.shape[0], 1))
                    S2A2 = np.append(self.S2, A2, axis=1)
                    Q2[:, a2] = Qfunctions[-1].predict(S2A2)
                max_Q2 = np.max(Q2, axis=1)
                value = self.R + gamma * (1 - self.D) * max_Q2
            Q = RandomForestRegressor()
            Q.fit(SA, value)
            Qfunctions.append(Q)
        return Qfunctions

    def greedy_action(self, s, nb_actions=10):
        """
        Return the greedy action with respect to the Q-function and the state s.
        """
        Qsa = []
        for a in range(nb_actions):
            sa = np.append(s, a).reshape(1, -1)
            Qsa.append(self.Q.predict(sa))
        return np.argmax(Qsa)

    def act(self, s):
        return self.greedy_action(s)
