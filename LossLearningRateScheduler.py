from __future__ import absolute_import
from __future__ import print_function

import keras
from keras import backend as K
import numpy as np

class LossLearningRateScheduler(keras.callbacks.Callback):
    
    ## decay_rate = 0.95
    def __init__(self, model, base_lr, lr_csv_path, decay_threshold = 0.005, decay_rate = 0.95, loss_type = 'loss'):

        super(LossLearningRateScheduler, self).__init__()
        self.model = model
        self.base_lr = base_lr
        self.decay_threshold = decay_threshold
        self.decay_rate = decay_rate
        self.loss_type = loss_type
        self.losses = []
        self.lr_csv_path = lr_csv_path
    
    def on_epoch_begin(self, epoch, logs=None):
        print("on_epoch_begin, epoch = ", epoch)
    
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        print("self.losses = ", self.losses)
        if(epoch >= 2):
            current_lr = K.get_value(self.model.optimizer.lr)

            target_loss = self.losses

            loss_diff =  target_loss[-2] - target_loss[-1]

            if loss_diff <= self.decay_threshold :

                print(' '.join(('Changing learning rate from', str(current_lr), 'to', str(current_lr * self.decay_rate))))
                K.set_value(self.model.optimizer.lr, current_lr * self.decay_rate)
                current_lr = current_lr * self.decay_rate

            if(epoch == 2):
                mode = 'w'
            else:
                mode = 'a'
            with open(self.lr_csv_path, mode) as f:
                f.write("epoch = %d, lr = %f\n" %(epoch, current_lr))
			


        return K.get_value(self.model.optimizer.lr)



