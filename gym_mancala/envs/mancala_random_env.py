import gym
import random
from gym_mancala.envs.mancala_env import MancalaEnv
from gym_mancala.envs.board import Board

class MancalaRandomEnv(MancalaEnv):


    def step(self, action):
        while int(self.board.player2_turn) != self.player:
            move = random.randint(0, 5)
            for i in range(20):
                move = random.randint(0, 5)
                if self.board.marbles[1][move] != 0:
                    break
            self.board.execute_turn(move)
        self.board.execute_turn(action)
        ob = self.board.marbles
        return ob, self.calculate_reward(), self.board.game_over, {}


