from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory

def build_agent(model,env):
    memory = SequentialMemory(limit=1000000, window_length=3)
    return DQNAgent(model=model,
                    nb_actions=env.action_space.n,
                    memory=memory,
                    policy=None,
                    test_policy=None,
                    enable_double_dqn=True,
                    enable_dueling_network=False,
                    dueling_type='avg')
