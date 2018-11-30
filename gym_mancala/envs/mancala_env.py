import gym
from gym_mancala.envs.board import Board

class MancalaEnv(gymEnv):

    def __init__(self):
        self.board = Board()

    def step(self, action):
        self.board.execute_turn(action)

    def reset(self):
        self.board = Board

    def render(self):
        self.board.print_board()

    def calculate_reward(self):
        return self.board.mancala[0] - self.board.mancala[0] + sum(self.board.marbles[0]) - sum(self.board.marbles[0])