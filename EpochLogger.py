from keras.callbacks import Callback


class EpochLogger(Callback):
    def __init__(self, test_data, num_epochs):
         self.test_data = test_data
         self.num_epochs = num_epochs
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print(f'Epoch {epoch}/{self.num_epochs} - Loss: {loss} - Acc: {acc}')
