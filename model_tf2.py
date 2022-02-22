# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


# variable initialization functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model(Model):
    def __init__(self, n_class=10):
        super(Model, self).__init__()
        self.dense1 = Dense(50,activation='relu')
        self.dense2 = Dense(n_class)
        # self.var_list = self.trainable_weights
        # self.y = self.dense2(self.dense1(x))
        # self.var_list = self.trainable_weights

        # in_dim = int(x.get_shape()[1]) # 784 for MNIST
        # out_dim = int(y_.get_shape()[1]) # 10 for MNIST
        
        # self.x = x # input placeholder

        # simple 2-layer network
        # W1 = weight_variable([in_dim,50])
        # b1 = bias_variable([50])

        # W2 = weight_variable([50,out_dim])
        # b2 = bias_variable([out_dim])

        # h1 = tf.nn.relu(tf.matmul(x,W1) + b1) # hidden layer
        # self.y = tf.matmul(h1,W2) + b2 # output layer

        # self.var_list = [W1, b1, W2, b2]
    
    def acc(self, x, y):
        self.y = self.call(x)
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return self.accuracy
    
    def call(self, x):
        x = self.dense1(x)
        return self.dense2(x)

    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.trainable_weights)):
            self.F_accum.append(np.zeros(self.trainable_weights[v].get_shape().as_list()))

        # sampling a random class from softmax
        # probs = tf.nn.softmax(self.y)
        # # probs = self.y
        # class_ind = tf.argmax(probs,1)[0]
        # class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        # fish_gra = tf.gradients(tf.math.log(probs[0,class_ind]), self.trainable_weights)

        fish_gra = get_fish()
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            # ders = sess.run(fish_gra, feed_dict={self.x: imgset[im_ind:im_ind+1]})
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
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.trainable_weights)):
            self.star_vars.append(np.zeros(self.trainable_weights[v].shape))
            self.star_vars[v] = deepcopy(self.trainable_weights[v].numpy())

    def restore(self):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.trainable_weights)):
                # self.trainable_weights[v] = tf.Variable(deepcopy(self.star_vars[v]))
                self.trainable_weights[v].assign(self.star_vars[v])

    def set_vanilla_loss(self,y):
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.y))
        # self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

    # def update_ewc_loss(self, lam):
    #     # elastic weight consolidation
    #     # lam is weighting for previous task(s) constraints

    #     if not hasattr(self, "ewc_loss"):
    #         self.ewc_loss = self.cross_entropy

    #     for v in range(len(self.trainable_weights)):
    #         self.ewc_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.trainable_weights[v] - self.star_vars[v])))
    #     self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.ewc_loss)
    
# TRAIN TEST MLP
def get_fish():
    @tf.function
    def train_fish(x, mod):
        with tf.GradientTape() as tape:
            y_out = tf.nn.softmax(mod(x,training=True))
            c_index = tf.argmax(y_out,1)[0]
            loss = tf.math.log(y_out[0,c_index])
            # loss = tf.keras.losses.categorical_crossentropy(y,y_out)

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
                print('hi')
                for v in range(len(mod.trainable_weights)):
                    loss += (lam/2) * tf.reduce_sum(tf.multiply(mod.F_accum[v].astype(np.float32),tf.square(mod.trainable_weights[v] - mod.star_vars[v])))
        gradients = tape.gradient(loss,mod.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mod.trainable_weights))
    
        train_loss(loss)
        train_accuracy(y, y_out)
    
    return train_step



