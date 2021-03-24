# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 23:51:27 2021

@author: kulka
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer

# creates a simple linear layer, lazily calls build when input it provided
class Linear(Layer):
    
    # units are the dimensions of the output vector
    def __init__(self, units=32):
        super().__init__()
        self.units = units
        
    # input shape is the dimensions of the row vector input, lazily called, (see
    # Linear_Test class below)
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.units), 
                                  initializer='random_normal', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='random_normal', 
                                  trainable=True)
    # notice that since all vectors are row vectors, the matrix multiplication
    # is reversed
    def call(self, inputs):
        return inputs @ self.w + self.b
    
# creates a linear layer
class Linear_Test:
    def __init__(self, units=32):
        self.ll = Linear(units)
    
    def output(self, inputs = tf.constant([[1.,2.,3.], [4.,5.,6.]])):
        return self.ll(inputs)

class SGD_MutiClass:
    # dataset in the form of an array of tuples
    def init(self, dataset, classes):
        self.dataset = dataset
        self.classes = classes
        
    def apply(self, learning_rate):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
        ll = Linear(self.classes)
        
        for step, (x,y) in enumerate(self.dataset):
            with tf.GradientTape() as tape:
                logits = ll(x)
                loss = loss_fn(y, logits)
                
                gradients = tape.gradient(loss, ll.trainable_weights)
                optimizer.apply_gradients(zip(gradients, ll.trainable_weights))
        return ll

x = tf.Variable(3.)


                