import os

# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import random
import math
import numpy as np

from gym_mancala.envs.mancala_random_env import MancalaRandomEnv
from agent import build_agent
from model import build_model
from keras.optimizers import Adam, SGD
from keras.models import load_model

MODEL_NUMBER = 7
NETWORKS_PATH = 'networks/'
PATH = NETWORKS_PATH + 'Model' + str(MODEL_NUMBER) + '/'
STEPS = 5000


def train_network():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    environment = MancalaRandomEnv()
    model = build_model(environment)
    model.save(PATH + 'model.HDF5')
    agent = build_agent(model, environment, STEPS)
    agent.compile(optimizer=Adam(lr=.01))
    agent.fit(environment,
              nb_steps=STEPS,
              action_repetition=1,
              callbacks=None,
              verbose=2,
              visualize=True,
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
            print('Testing: ' + filename)
            if 'HDF5' not in filename:
                agent = build_agent(model, environment, STEPS)
                agent.compile(optimizer=Adam(lr=.01))
                agent.load_weights(NETWORKS_PATH + dirname + '/' + filename)
                try:
                    history = agent.test(environment, nb_episodes=10,
                               action_repetition=1,
                               callbacks=None
                               , visualize=False,
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
    print("TODO")


if __name__ == '__main__':
    "Welcome to MancalaAI!"
    userInput = input("Would you like to [t]rain a network, test [n]etworks, or [p]lay against a network?")
    userInput = userInput.lower()
    if userInput == 't':
        train_network()
    elif userInput == 'n':
        test_networks()
    elif userInput == 'p':
        networkPath = input("Specify the network path or leave blank to play against the best network:")
        play_network()
    else:
        print("Invalid input")
        print("Please enter t, n, or p")
