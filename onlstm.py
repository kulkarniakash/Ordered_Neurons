# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:14:58 2021

@author: kulka
"""
import numpy as np
from numpy.random import default_rng
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1 import placeholder

filename = "aesops_fable.txt"

def read_file(fn):
    with open(fn, encoding='utf_8') as file:
        content = file.readlines()
    content = np.array([word for c in content for word in c.strip('\n').split()])
    content = content.reshape(-1)
    content = [word.strip() for word in content]
    content = np.array(content)
    return content

def build_dictionary(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dict = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dict

words = read_file(filename)
dictionary, reverse_dict = build_dictionary(words)
doc_size = len(words)
vocab_size = len(dictionary)
n_hidden = 512
emb_dim = 6
n_input = 3


rnn_units = n_hidden
input_dim = emb_dim
batch_size = 1
time_step = 3
sequence_len = 3
vocab_size = 5
def build_dataset(content):
    x = []
    y = []
    for i in range(doc_size - n_input):
        step_words = words[i : i + n_input]
        label = dictionary[words[i+n_input]]
        seq_words = []
        for word in step_words:
            seq_words.append(dictionary[word])
        seq_words = np.array(seq_words)
        x.append(seq_words)
        y.append(label)
    x = np.array(x)
    y = np.array(y)
    return x, y

# Test gate layer!!
# assumes batch size is 1!!
class GateLayer(layers.Layer):
    
    # activation functions expected to be a function object 
    def __init__(self, activation_func, hidden_units=100):
        super().__init__()
        self.units = hidden_units
        self.activation_func = activation_func
        
    def build(self, input_shape):
        self.input_weights = self.add_weight(shape=(input_shape[-1] - self.units, self.units),
                                             initializer="random_normal", 
                                             trainable=True)
        self.hidden_weights = self.add_weight(shape=(self.units, self.units), 
                                              initializer="random_normal",
                                              trainable=True)
        self.bias = self.add_weight(shape=(self.units,), initializer="random_normal", 
                                    trainable=True)
        
    def call(self, inputs):
        x_t = inputs[0, :inputs.shape[-1] - self.units]
        h_previous = inputs[0, inputs.shape[-1] - self.units:]
        
        return self.activation_func(tf.tensordot(x_t, self.input_weights, axes=1) 
                                    + tf.tensordot(h_previous, self.hidden_weights, axes=1)
                                    + self.bias)
    
class ONLSTM_Cell(layers.Layer):
    
    def __init__(self, units):
        self.units = units
        self.state_size = [units, units]
        self.output_size = units
        super().__init__()
        
    # @property
    # def state_size(self):
    #   return [self.units, self.units]
  
    def build(self, input_shape):
        sig = tf.nn.sigmoid
        tanh = tf.nn.tanh 
        
        self.f_gate = GateLayer(sig, self.units)
        self.i_gate = GateLayer(sig, self.units)
        self.o_gate = GateLayer(sig, self.units)
        self.c_hat_gate = GateLayer(tanh, self.units)
        
        self.f_tilde = GateLayer(cumax, self.units)
        self.i_tilde_mod = GateLayer(cumax, self.units)
        
        self.built = True
    
    def get_initial_state(inputs):
        return [tf.zeros([batch_size, rnn_units]), tf.zeros([batch_size, rnn_units])]
    
    def call(self, inputs, states):
        tanh = tf.nn.tanh
        
        if states == None:
            states = [tf.zeros(self.units), tf.zeros(self.units)]
        
        hidden_prev = states[0]
        cell_prev = states[1]
        concat_input = tf.concat([inputs, hidden_prev], -1)
        
        f = self.f_gate(concat_input)
        i = self.i_gate(concat_input)
        o = self.o_gate(concat_input)
        c_hat = self.c_hat_gate(concat_input)
        
        f_tilde = self.f_tilde(concat_input)
        i_tilde = 1 - self.i_tilde_mod(concat_input)
        
        omega = f_tilde * (1 - i_tilde)
        f_hat = f * omega + (f_tilde - omega)
        i_hat = i * omega + (i_tilde - omega)
        
        cell = f_hat * cell_prev + i_hat * c_hat
        hidden = o * tanh(cell)
        
        return hidden, [hidden, cell]

# assumes batch size is 1!!
# @tf.function
def cumax(array):
    # array = to_numpy(tf.nn.softmax(array))
    # ans = []
    # sumnow = 0
    # for j in range(array.shape[0]):
    #     sumnow += array[j]
    #     ans.append(sumnow)
    return tf.cumsum(tf.nn.softmax(array))

def to_numpy(tfarr):
    return tfarr.numpy()

def test():
    a = [1,2,3]
    return tf.constant(a)





t = tf.constant([1,2,3]) 
t =  tf.reshape(t, [batch_size, sequence_len])



rng = default_rng()
initial_state =  [tf.zeros([batch_size, rnn_units]), tf.zeros([batch_size, rnn_units])]

states = [tf.zeros(rnn_units), tf.zeros(rnn_units)]
inputs = keras.Input((sequence_len))
# take in [batch_size, input_length] where input_length is the length of a sequence of 
# inputs
embed = layers.Embedding(vocab_size, input_dim)
# outputs shape of (batch_size, input_length, output_dim)
v = embed(inputs)
#[batch_size, time_steps, input_dim]
onlstm_outputs = layers.RNN(ONLSTM_Cell(rnn_units))(v, initial_state=initial_state)
output = layers.Dense(vocab_size, activation="softmax")

model = keras.Model(inputs=inputs, outputs=output)
print(model.predict(t))

# n_gram = keras.Input(shape=(n_input, ))
# word_emb = layers.Embedding(vocab_size, emb_dim)(n_gram)
# output_gate = layer.Dense()
# y = np.array([1,2,3]).reshape(-1, 1, 3).astype('float32')
# _, state_h, state_c = layers.LSTM(10, return_state=True)(y)
