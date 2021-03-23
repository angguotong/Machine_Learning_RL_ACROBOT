import tensorflow as tf
import numpy as np
import gym
import random
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import tensorflow.keras.backend as K
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt


class Acrobot_DQN_Decent_Gradient():
    def __init__(self):
        self.env = gym.make("Acrobot-v1")
        self.ep = 0.01
        self.ep_min = 0.01
        self.ep_decay = 0.99
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(0.0005)
        self.memory = deque (maxlen = 50_000)
        self.loss_memory = deque(maxlen = 3500)
        self.reward_memory = []
        self.toggle_key = False


    def model (self):
        Inputs = (Input(shape = self.env.observation_space.shape[0]))
        Layer1 = (Dense(32, activation = "relu"))(Inputs)
        Layer2 = (Dense(64, activation = "relu"))(Layer1)
        Layer3 = (Dense(32, activation = "relu"))(Layer2)
        Outputs = (Dense(self.env.action_space.n, activation = "linear")) (Layer3)
        return tf.keras.Model(inputs = Inputs, outputs = Outputs)


    def Train (self, memory, model, target_model, Counter):
        batch_size = 32
        discount_factor = 0.95
        Counter = 0
        if len(memory) <= batch_size:
            return

        mini_batch_size = random.sample(memory, batch_size)
        s_ = []
        action_ = []
        updated_q_r_list = []
        for s,s1,r,d,action in mini_batch_size:
            s1 = s1.reshape(1,self.env.observation_space.shape[0])
            if d:
                updated_q_r = r
            else:
                a = model.predict(s1)[0]
                t = target_model.predict(s1)[0]
                updated_q_r = r + discount_factor * t[np.argmax(a)]
        
            one_hot = tf.one_hot(np.array(action), self.env.action_space.n)

            with tf.GradientTape() as tape:
                predicted_q_value = model(s)
                predicted_q_value = tf.reduce_sum(tf.multiply(predicted_q_value, one_hot), axis = 1)
                loss = self.loss_function(updated_q_r,predicted_q_value)
            self.loss_memory.append(loss)
            gradient = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradient, model.trainable_variables))


    def main(self):
        Training_Episode = 100_000
        Training_Counter = 0
        model = self.model()
        target_model = self.model()
        for Train in range(Training_Episode):
            plt.clf()
            Counter = 0
            s = self.env.reset()
            Episode = 600
            for score in range (Episode):
                s = s.reshape(1,self.env.observation_space.shape[0])

                if random.random() <= self.ep:
                    action = self.env.action_space.sample()
                else:
                    action_ = model(s, training = False) #This will return me a tf numpy
                    action = tf.argmax(action_[0]).numpy() #This will change return me 1 single value

                s1,r,d,_ = self.env.step(action)

                self.memory.append([s,s1,r,d,action])

                if d:
                    if r == 0:
                        r = 100
                s = s1
                Training_Counter += 1
                Counter += 1
                if Training_Counter % 4 == 0:
                    self.Train(self.memory, model, target_model, Counter)
                if d:
                    self.reward_memory.append(score * -1)
                    print ("Training Ep {} , Score {}, Ep {}".format(Train,score*-1,self.ep))
                    target_model.set_weights(model.get_weights())
                    if self.ep> self.ep_min:
                        self.ep *= (self.ep_decay) # reduce greed function
                break
Acrobot_DQN_Decent_Gradient()