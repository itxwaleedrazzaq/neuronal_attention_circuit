import gym
import numpy as np

from keras._tf_keras.keras.layers import (
    Input,
    Dense,
    Flatten,
    Dropout,
    TimeDistributed,
    Conv2D
    )
from keras._tf_keras.keras.models import Model

from liquid_attention import LAN


# Configuration
STEPS = 2000
MODEL_NAME = "CarRacing_LAN"
WEIGHTS_DIR = "model_weights"


# Model definition
def build_model(input_shape=(None, 96, 96, 3)):
    inp = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(10, (3, 3), strides=2, activation="relu"))(inp)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(20, (5, 5), strides=2, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(30, (5, 5), strides=2, activation="relu"))(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Flatten())(x)
    x = LAN(
        d_model=64,
        num_heads=16,
        mode="exact",
        return_sequences=False
    )(x)

    x = Dense(64, activation="relu")(x)
    out = Dense(5, activation="softmax")(x)

    return Model(inputs=inp, outputs=out)


# Main execution loop
def main():
    env = gym.make("CarRacing-v1", continuous=False)

    model = build_model()
    model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.weights.h5")

    obs = env.reset()
    while True:
        for _ in range(STEPS):
            env.render()
            obs_input = np.expand_dims(np.expand_dims(np.array(obs), axis=0), axis=1)
            predicted_action = model.predict(obs_input, verbose=0).squeeze()
            obs, _, _, _ = env.step(np.argmax(predicted_action))

        obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
