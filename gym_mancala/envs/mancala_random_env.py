import gym
import random
from gym_mancala.envs.mancala_env import MancalaEnv
from gym_mancala.envs.board import Board

class MancalaRandomEnv(MancalaEnv):


    def step(self, action):
        self.board.execute_turn(action)
        move = random.randint(0, 5)
        while(self.board.marbles[1][move] == 0):
            move = random.randint(0, 5)
        self.board.execute_turn(move)
        ob = self.board.marbles
        return ob, self.calculate_reward(), self.board.game_over, {}


