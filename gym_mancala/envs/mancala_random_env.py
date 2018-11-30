import gym
import random
from gym_mancala.envs.mancala_env import MancalaEnv
from gym_mancala.envs.board import Board

class MancalaRandomEnv(MancalaEnv):

    def __init__(self):
        self.board = Board()

    def step(self, action):
        self.board.execute_turn(action)
        move = random.randint(1, 6)
        while(self.board.marbles[move] == 0):
            move = random.randint(1,6)
        self.board.execute_turn(move)
        ob = self.env.getState()
        return ob, self.calculate_reward(), self.board.game_over, {}


    def reset(self):
        self.board = Board

    def render(self):
        self.board.print_board()

