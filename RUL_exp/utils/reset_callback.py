from keras._tf_keras.keras.callbacks import Callback
from keras._tf_keras.keras.layers import LSTM

class ResetStatesCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:  
            if isinstance(layer, (LSTM)) and layer.stateful:  # Reset only stateful RNN layers
                layer.reset_states()
