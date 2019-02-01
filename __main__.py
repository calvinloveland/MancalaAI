import os

os.environ["KERAS_BACKEND"] = "theano"
import random
import math
import numpy as np

from gym_mancala.envs import MancalaUserEnv
from gym_mancala.envs.board import Board
from gym_mancala.envs.mancala_random_env import MancalaRandomEnv
from agent import build_agent
from model import build_model
from keras.optimizers import Adam, SGD
from keras.models import load_model
from shared.priority import set_background_priority

MODEL_NUMBER = 9
NETWORKS_PATH = 'networks/'
PATH = NETWORKS_PATH + 'Model' + str(MODEL_NUMBER) + '/'
BEST_NETWORK_MODEL = NETWORKS_PATH + 'Model2/model.HDF5'
BEST_NETWORK_WEIGHTS = NETWORKS_PATH + 'Model2/1881'
STEPS = 20000000


def train_network():
    set_background_priority()

    if not os.path.exists(PATH):
        os.makedirs(PATH)
    environment = MancalaRandomEnv()
    model = build_model(environment)
    model.save(PATH + 'model.HDF5')
    agent = build_agent(model, environment, STEPS)
    agent.compile(optimizer=Adam(lr=.1))
    agent.fit(environment,
              nb_steps=STEPS,
              action_repetition=1,
              callbacks=None,
              verbose=2,
              visualize=False,
              nb_max_start_steps=0,
              start_step_policy=None,
              log_interval=math.floor(STEPS / 10),
              nb_max_episode_steps=None)
    network_id = random.randint(1, 10000)
    agent.save_weights(PATH + str(network_id))
    print("Saved network: " + str(network_id))


def test_networks():
    avg_scores = {}
    dirnames = os.listdir(NETWORKS_PATH)
    for dirname in dirnames:
        filenames = os.listdir(NETWORKS_PATH + dirname)
        environment = MancalaRandomEnv()
        model = load_model(NETWORKS_PATH + dirname + "/model.HDF5")
        print(model.summary())
        for filename in filenames:
            if 'HDF5' not in filename:
                print('Testing: ' + dirname + '/' + filename)
                agent = build_agent(model, environment, STEPS)
                agent.compile(optimizer=Adam(lr=.01))
                agent.load_weights(NETWORKS_PATH + dirname + '/' + filename)
                try:
                    history = agent.test(environment, nb_episodes=100,
                                         action_repetition=1,
                                         callbacks=None,
                                         visualize=False,
                                         nb_max_episode_steps=None,
                                         nb_max_start_steps=0,
                                         start_step_policy=None,
                                         verbose=2)
                    avg = np.mean(history.history.get('episode_reward'))
                    print("Average score: " + str(avg))
                    avg_scores[dirname + '/' + filename] = avg
                except:
                    print("Invalid format: " + filename)
    max = -1000
    best = None
    for key in avg_scores.keys():
        if avg_scores.get(key) > max:
            best = key
            max = avg_scores.get(key)
    print("Best network = " + best + " with avg of: " + str(max))


def play_network():
    print("You are Player 2 on the bottom of the board")
    print("When prompted give a space between 0-5")
    print("Selecting an empty space will cause you to lose the game")
    model = load_model(BEST_NETWORK_MODEL)
    environment = MancalaUserEnv()
    environment.board.print_board()
    agent = build_agent(model, environment, STEPS)
    agent.compile(optimizer=Adam(lr=.01))
    agent.load_weights(BEST_NETWORK_WEIGHTS)
    agent.test(environment, nb_episodes=1,
               action_repetition=1,
               callbacks=None,
               visualize=False,
               nb_max_episode_steps=None,
               nb_max_start_steps=0,
               start_step_policy=None,
               verbose=0)


if __name__ == '__main__':
    "Welcome to MancalaAI!"
    userInput = input("Would you like to [t]rain a network, test [n]etworks, or [p]lay against a network?")
    userInput = userInput.lower()
    if userInput == 't':
        train_network()
    elif userInput == 'n':
        test_networks()
    elif userInput == 'p':
        play_network()
    else:
        print("Invalid input")
        print("Please enter t, n, or p")
