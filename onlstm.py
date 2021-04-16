# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 20:14:58 2021

@author: kulka
"""
from nltk.tokenize import RegexpTokenizer
import numpy as np
from numpy.random import default_rng
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.compat.v1 import placeholder
from tensorflow.keras.callbacks import LambdaCallback

filename = "aesops_fable.txt"

def read_file(fn):
    with open(fn, mode='r', encoding='utf_8') as file:
        content = file.read()
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(content)

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
emb_dim = 10


rnn_units = n_hidden
input_dim = emb_dim
batch_size = 1
time_step = 3
sequence_len = 7

def build_dataset(content):
    x = []
    y = []
    for i in range(doc_size - sequence_len):
        step_words = words[i : i + sequence_len]
        label = dictionary[words[i+sequence_len]]
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
# class GateLayer(layers.Layer):
    
#     # activation functions expected to be a function object 
#     def __init__(self, activation_func, hidden_units=100):
#         super().__init__()
#         self.units = hidden_units
#         self.activation_func = activation_func
        
#     def build(self, input_shape):
#         self.input_weights = self.add_weight(shape=(input_shape[-1] - self.units, self.units),
#                                              initializer="random_normal", 
#                                              trainable=True)
#         self.hidden_weights = self.add_weight(shape=(self.units, self.units), 
#                                               initializer="random_normal",
#                                               trainable=True)
#         self.bias = self.add_weight(shape=(self.units,), initializer="random_normal", 
#                                     trainable=True)
        
    
#     def call(self, inputs):
#         x_t = inputs[0, :(inputs.shape[-1] - self.units)]
#         h_previous = inputs[0, (inputs.shape[-1] - self.units):]
        
#         return self.activation_func(tf.tensordot(x_t, self.input_weights, axes=1) 
#                                     + tf.tensordot(h_previous, self.hidden_weights, axes=1)
#                                     + self.bias)

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
        # self.built = True
        
    @tf.function
    def call(self, inputs):
        print("GInput shape", inputs[0])
        x_t = inputs[0, :(inputs.shape[-1] - self.units)]
        h_previous = inputs[0, (inputs.shape[-1] - self.units):]
        
        return self.activation_func(tf.tensordot(x_t, self.input_weights, axes=1) 
                                    + tf.tensordot(h_previous, self.hidden_weights, axes=1)
                                    + self.bias)
    
class ONLSTM_Cell(layers.Layer):
    
    def __init__(self, units):
        self.units = units
        self.state_size = [units, units]
        self.output_size = units
        # sig = tf.nn.sigmoid
        # tanh = tf.nn.tanh 
        
        # self.f_gate = tf.function(GateLayer(sig, self.units))
        # self.i_gate = tf.function(GateLayer(sig, self.units))
        # self.o_gate = tf.function(GateLayer(sig, self.units))
        # self.c_hat_gate = tf.function(GateLayer(tanh, self.units))
        
        # self.f_tilde = tf.function(GateLayer(cumax, self.units))
        # self.i_tilde_mod = tf.function(GateLayer(cumax, self.units))
        super().__init__()
        
    # @property
    # def state_size(self):
    #   return [self.units, self.units]
  

    def build_gate_weights(self, input_shape, activation_func):
         input_weights = self.add_weight(shape=(input_shape[-1], self.units),
                                              initializer="random_normal", 
                                              trainable=True)
         hidden_weights = self.add_weight(shape=(self.units, self.units), 
                                              initializer="random_normal",
                                              trainable=True)
         bias = self.add_weight(shape=(self.units,), initializer="random_normal", 
                                    trainable=True)
        
         return input_weights, hidden_weights, bias, activation_func
    

    # inputs.shape = (None, units + emb_dim)
    def call_gate(self, gate, inputs):
        input_weights, hidden_weights, bias, activation_func = gate
        x_t = inputs[:, :(inputs.shape[-1] - self.units)]
        h_previous = inputs[:, (inputs.shape[-1] - self.units):]
        
        return activation_func(tf.tensordot(x_t, input_weights, axes=1) 
                                    + tf.tensordot(h_previous, hidden_weights, axes=1)
                                    + bias)
    
    def build(self, input_shape):
        sig = tf.nn.sigmoid
        tanh = tf.nn.tanh 
        self.f_gate = self.build_gate_weights(input_shape, sig)
        self.i_gate = self.build_gate_weights(input_shape, sig)
        self.o_gate = self.build_gate_weights(input_shape, sig)
        self.c_hat_gate = self.build_gate_weights(input_shape, tanh)
        
        self.f_tilde = self.build_gate_weights(input_shape, cumax)
        self.i_tilde_mod = self.build_gate_weights(input_shape, cumax)
        
        self.built = True
        
    
    # def get_initial_state():
    #     return tf.constant([tf.zeros([batch_size, rnn_units]), tf.zeros([batch_size, rnn_units])])
    
    # inputs.shape = (None, emb_dim)
    def call(self, inputs, states):
        tanh = tf.nn.tanh
        
        # # is it processing all the batches though?
        # inputs = tf.reshape(inputs, (-1,))
        # print("Input Shape", inputs[2])
        # print("State Shape", tf.shape(states[0]))
        # inputs = tf.reshape(inputs, (-1, inputs.shape[-1]))
        
        if states == None:
            states = [tf.zeros(self.units), tf.zeros(self.units)]
            
        hidden_prev = states[0]
        cell_prev = states[1]
        concat_input = tf.concat([inputs, hidden_prev], -1)
        
        f = self.call_gate(self.f_gate, concat_input)
        i = self.call_gate(self.i_gate, concat_input)
        o = self.call_gate(self.o_gate, concat_input)
        c_hat = self.call_gate(self.c_hat_gate, concat_input)
        
        f_tilde = self.call_gate(self.f_tilde, concat_input)
        i_tilde = 1 - self.call_gate(self.i_tilde_mod, concat_input)
        
        omega = f_tilde * (i_tilde)
        f_hat = f * omega + (f_tilde - omega)
        i_hat = i * omega + (i_tilde - omega)
        
        cell = f_hat * cell_prev + i_hat * c_hat
        hidden = o * tanh(cell)
        
        return hidden, [hidden, cell]

# assumes batch size is 1!!
# @tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.float32),), 
#               experimental_relax_shapes=True)
def cumax(array):
    return tf.cumsum(tf.nn.softmax(array))


def get_class_label(array):
    return np.argmax(array)

# t = tf.constant([1,2.,3]) 
# t =  tf.reshape(t, [batch_size, sequence_len])
# t = t.numpy()
rng = default_rng()
x_train, y_train = build_dataset(words)
# [samples, time_step, feature]
x_train = x_train.reshape((-1, sequence_len)).astype('float32')
# y_train = y_train.reshape((-1, y_train.shape[0])).T

# y_train = y_train.reshape(-1, y_train.shape[0])
# x_train = x_train.reshape(-1, sequence_len, 1).astype('float32')

rng = default_rng()
# initial_state =  [tf.zeros([batch_size, rnn_units]), tf.zeros([batch_size, rnn_units])]

states = [tf.constant(rng.uniform(-1, 1, rnn_units)), tf.constant(rng.uniform(-1, 1, rnn_units))]
inputs = keras.Input((sequence_len))
# take in [batch_size, input_length] where input_length is the length of a sequence of 
# inputs
embed = layers.Embedding(vocab_size, input_dim, name="embed")
# outputs shape of (batch_size, input_length, output_dim)
v = embed(inputs)
#[batch_size, time_steps, input_dim]
onlstm_outputs = layers.RNN(ONLSTM_Cell(rnn_units),name="onlstm")(v)
output = layers.Dense(vocab_size, activation="softmax", name="output")(onlstm_outputs)
model = keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy())
# print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
model.fit(x_train, y_train, epochs=150, verbose=2)



def predict_next(input_words):
    input_words = np.array([dictionary[word] for word in input_words])
    input_words = input_words.reshape(-1, sequence_len, 1)
    res = np.array(model.predict(input_words))
    res = res.reshape(vocab_size)
    return reverse_dict[tf.math.argmax(res).numpy()]

print(predict_next(["the", "mice", "had", "a", "general", "council", "to"]))

# print(model.predict_class(t))

# n_gram = keras.Input(shape=(n_input, ))
# word_emb = layers.Embedding(vocab_size, emb_dim)(n_gram)
# output_gate = layer.Dense()
# y = np.array([1,2,3]).reshape(-1, 1, 3).astype('float32')
# _, state_h, state_c = layers.LSTM(10, return_state=True)(y)
