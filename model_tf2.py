import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras import Model

## Encoders
class MLPenc(Model):
    def __init__(self, latent_dim=4, name='enc'):
        super(MLPenc, self).__init__(name=name)
        self.dense1 = Dense(246, activation='relu')
        # self.bn1 = BatchNormalization()
        self.dense2 = Dense(128, activation='relu')
        # self.bn2 = BatchNormalization()
        self.dense3 = Dense(16, activation='relu')
        # self.bn3 = BatchNormalization()
        self.latent = Dense(latent_dim, activity_regularizer=tf.keras.regularizers.l1(10e-5))
        # self.bn4 = BatchNormalization()

    def call(self, x):
        x = self.dense1(x)
        # x = self.bn1(x)
        x = self.dense2(x)
        # x = self.bn2(x)
        x = self.dense3(x)
        # x = self.bn3(x)
        x = self.latent(x)
        return self.bn4(x)

## Classifier
class CLF(Model):
    def __init__(self, n_class=7, name='clf'):
        super(CLF, self).__init__(name=name)
        self.dense1 = Dense(n_class)

    def call(self, x):
        return self.dense1(x)

class Model(Model):
    def __init__(self, n_class=10):
        super(Model, self).__init__()
        self.enc = MLPenc()
        self.clf = CLF(n_class=n_class)
    
    def acc(self, x, y):
        self.y = self.call(x)
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.accuracy
    
    def call(self, x):
        x = self.enc(x)
        return self.clf(x)

    def compute_fisher(self, imgset, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.trainable_weights)):
            self.F_accum.append(np.zeros(self.trainable_weights[v].get_shape().as_list()))


        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        fish_gra = get_fish()
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = fish_gra(imgset[im_ind:im_ind+1],self)
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)

        plt.plot(range(0, num_samples-disp_freq, disp_freq), mean_diffs)
        plt.xlabel("Number of samples")
        plt.ylabel("Mean absolute Fisher difference")
        display.display(plt.gcf())
        display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples
        
        if not hasattr(self,"F_old"):
            self.F_old = deepcopy(self.F_accum)
            self.int = 1
        else:  
            self.int += 1
            for v in range(len(self.F_accum)):
                self.F_accum[v] = (self.F_accum[v] + self.F_old[v])/self.int
            self.F_old = deepcopy(self.F_accum)


    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.trainable_weights)):
            self.star_vars.append(deepcopy(self.trainable_weights[v].numpy()))

    def restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.trainable_weights)):
                self.trainable_weights[v].assign(self.star_vars[v])
    
# TRAIN TEST MLP
def get_fish():
    @tf.function
    def train_fish(x, mod):
        with tf.GradientTape() as tape:
            y_out = tf.nn.softmax(mod(x,training=True))
            c_index = tf.argmax(y_out,1)[0]
            loss = tf.math.log(y_out[0,c_index])

        gradients = tape.gradient(loss,mod.trainable_weights)
        return gradients
    return train_fish

def get_train_ewc():
    @tf.function
    def train_step(x, y, mod, optimizer, train_loss, train_accuracy, lam = 0):
        with tf.GradientTape() as tape:
            y_out = mod(x,training=True)
            loss = tf.keras.losses.categorical_crossentropy(y,y_out,from_logits=True)
            if hasattr(mod, "F_accum"):
                for v in range(len(mod.trainable_weights)):
                    loss += (lam/2) * tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v])))
        gradients = tape.gradient(loss,mod.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mod.trainable_weights))
    
        train_loss(loss)
        train_accuracy(y, y_out)
    
    return train_step



