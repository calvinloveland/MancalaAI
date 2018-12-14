from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def build_agent(model, env, steps):
    memory = SequentialMemory(limit=10000, window_length=1)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.25, value_min=.001, value_test=.05,
                                  nb_steps=steps)
    return DQNAgent(model=model,
                    nb_actions=env.action_space.n,
                    memory=memory,
                    policy=policy,
                    test_policy=None,
                    enable_double_dqn=True,
                    enable_dueling_network=False,
                    dueling_type='avg')
