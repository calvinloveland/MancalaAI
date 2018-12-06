import gym
from gym import spaces
from gym_mancala.envs.board import Board
import numpy as np


class MancalaEnv(gym.Env):

    def __init__(self):
        self.board = Board()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2, 6))
        self.player = 0

    def step(self, action):
        self.board.execute_turn(action)

    def reset(self):
        self.board = Board()
        obs = self.normalize_marbles()
        return obs

    def render(self, mode=None, close=None):
        self.board.print_board()

    def calculate_reward(self):
        return self.board.mancala[self.player] - self.board.mancala[1 - self.player] + sum(self.board.marbles[self.player]) - sum(self.board.marbles[1 - self.player])

    def normalize_marbles(self):
        return np.divide(np.subtract(self.board.marbles, 3), 3)
