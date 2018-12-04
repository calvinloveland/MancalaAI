import gym
from gym import spaces
from gym_mancala.envs.board import Board


class MancalaEnv(gym.Env):

    def __init__(self):
        self.board = Board()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(low=0, high=46, shape=(2, 6))

    def step(self, action):
        self.board.execute_turn(action)

    def reset(self):
        self.board = Board()
        obs = self.board.marbles
        return obs

    def render(self, mode=None, close=None):
        self.board.print_board()

    def calculate_reward(self):
        return self.board.mancala[0] - self.board.mancala[0] + sum(self.board.marbles[0]) - sum(self.board.marbles[0])