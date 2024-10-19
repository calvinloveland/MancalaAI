from tinygrad.nn import Sequential, Linear, Sigmoid


def build_model(env):
    inputShape = (1,) + env.observation_space.shape
    print("InputShape:")
    print(inputShape)
    model = Sequential([
        Linear(inputShape[1], 32), Sigmoid(),
        Linear(32, 64), Sigmoid(),
        Linear(64, 128), Sigmoid(),
        Linear(128, 256), Sigmoid(),
        Linear(256, 512), Sigmoid(),
        Linear(512, env.action_space.n), Sigmoid()
    ])
    print(model)
    return model
