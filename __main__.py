import os

#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from gym_mancala.envs.mancala_random_env import MancalaRandomEnv
from agent import build_agent
from model import build_model
from keras.optimizers import Adam



if __name__ == '__main__':
    environment = MancalaRandomEnv()
    model = build_model(environment)
    agent = build_agent(model, environment)
    agent.compile( optimizer=Adam(lr=.00025))
    agent.fit(environment,
              nb_steps=1000,
              action_repetition=1,
              callbacks=None,
              verbose=1,
              visualize=True,
              nb_max_start_steps=0,
              start_step_policy=None,
              log_interval=10000,
              nb_max_episode_steps=None)