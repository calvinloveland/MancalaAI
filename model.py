from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
import keras.backend as K

def build_model(env):

    inputShape = (3,) + env.observation_space.shape
    print("InputShape:")
    print(inputShape)
    model = Sequential()
    model.add(Dense(32, input_shape=inputShape, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(env.action_space.n, activation='linear'))
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='sgd',
    #               metrics=['accuracy'])
    print(model.summary())
    return model
