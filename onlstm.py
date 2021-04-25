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
from tensorflow.keras.callbacks import Callback
import pickle

sos = "<sos>"
eos = "<eos>"
unk = "<unk>"

class Perplexity:
    # ModelHanlder or model
    def __init__(self, model_handler=None, model_trained=None):
        if model_handler != None:
            self.model_handler = model_handler
            self.model = model_handler.model
        if model_trained != None:
            self.model = model_trained
        
        self.dictionary = model_handler.data_handler.dictionary
        self.reverse_dictionary = {self.dictionary[word] : word for word in self.dictionary}
    
    # takes in a data handler object
    def calculate_bigram_perp(self, test_data_handler, max_sentences):
        self.perp = 1
        test_data = test_data_handler.get_text_list(max_sentences)
        test_data_handler.build_dictionary(test_data)
        test_dictionary = test_data_handler.dictionary
        
        known_words = set(self.dictionary.keys()).intersection(set(test_dictionary.keys()))
        
        inputs = np.array([self.dictionary[test_data[i-1]] for i in range(1, len(test_data)) 
                  if test_data[i-1] in self.dictionary and test_data[i] in self.dictionary
                          and test_data[i-1] != eos])
        labels = np.array([self.dictionary[test_data[i]] for i in range(1, len(test_data))
                  if test_data[i] in self.dictionary and test_data[i-1] in self.dictionary
                          and test_data[i] != sos])
        
        inputs = inputs.reshape(-1, 1)
        distributions = self.model.predict(inputs)
        
        for i, dist in enumerate(distributions):
            self.perp *= (1 / dist[labels[i]]) ** (1 / len(test_data))
            
        # self.perp = self.perp ** (1 / len(test_data))
        self.known_ratio = len(known_words) / len(test_dictionary)
        
        return self.perp, self.known_ratio
        
class DataHandler:
    def __init__(self, fn):
        self.fn = fn
        
    # reads text, inserts sos and eos wherever necessary and returns list of list of words
    def read_lines(self, fn):
        with open(fn, mode='r', encoding='utf_8') as file:
            content = file.readlines()
        content = [' '.join([sos, sent, eos]) for sent in content]
        return [sent.split() for sent in content]
    
    # takes in a list of list of words and returns a list of words
    def flatten_sentences(self, sents):
        whole_text = []
        for sent in sents:
            whole_text = whole_text + sent
        return whole_text
    
    # takes in list of words and builds a dictionary labelled sequentially starting from 1
    def build_dictionary(self, words):
        count = collections.Counter(words).most_common()
        self.dictionary = dict()
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary) + 1
        self.reverse_dict = dict(zip(self.dictionary.values(), self.dictionary.keys()))
    
    def get_text_list(self, max_sentences):
        text = self.read_lines(self.fn)
        text = text[:max_sentences]
        text = self.flatten_sentences(text)
        
        return text
    
    # returns the training data and labels
    def build_dataset(self, max_sentences, sequence_len):
        x = []
        y = []
        self.sequence_len = sequence_len
        self.train_text = self.get_text_list(max_sentences)
        self.build_dictionary(self.train_text)
        
        for i in range(len(self.train_text) - sequence_len):
            step_words = self.train_text[i : i + sequence_len]
            label = self.dictionary[self.train_text[i+sequence_len]]
            seq_words = []
            for word in step_words:
                seq_words.append(self.dictionary[word])
            seq_words = np.array(seq_words)
            x.append(seq_words)
            y.append(label)
        x = np.array(x)
        y = np.array(y)
        return x, y

class ModelHandler:
    # takes in a DataHandler object, word embedding dimension, hidden units
    def __init__(self, data_handler, embedding_dim, units, dh_val):
        self.data_handler = data_handler
        self.emb_dim = embedding_dim
        self.h_units = units
        self.dh_val = dh_val
        self.model = None
        
    # regularizer = "l1" or "l2"
    def buildModelONLSTM(self, max_sentences, epochs):
        self.sequence_len = dh.sequence_len
        self.x_train, self.y_train = self.data_handler.build_dataset(max_sentences, self.sequence_len)
        dictionary, reverse_dictionary = self.data_handler.dictionary, self.data_handler.reverse_dict
        vocab_size = len(dictionary)
        
        perp_callback = PrintPerplexity(self, self.dh_val, max_sentences)
        
        rng = default_rng()
        states = [tf.constant(rng.uniform(-1, 1, self.h_units)), tf.constant(rng.uniform(-1, 1, self.h_units))]
        # inputs = keras.Input((self.sequence_len))
        
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(vocab_size + 1, self.emb_dim, name="embed"))
        self.model.add(layers.RNN(ONLSTM_Cell(self.h_units),name="onlstm"))
        self.model.add(layers.Dense(vocab_size+1, activation="softmax", name="output"))
        self.x_train = x_train.reshape((-1, self.sequence_len)).astype('int32')
        
        # # take in [batch_size, input_length] where input_length is the length of a sequence of 
        # # inputs
        # embed = layers.Embedding(vocab_size+1, self.emb_dim, name="embed")
        # # outputs shape of (batch_size, input_length, output_dim)
        # v = embed(inputs)
        # #[batch_size, time_steps, input_dim]
        # onlstm_outputs = layers.RNN(ONLSTM_Cell(self.h_units),name="onlstm")(v)
        # output = layers.Dense(vocab_size+1, activation="softmax", name="output")(onlstm_outputs)
        # self.model = keras.Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy())
        # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
        self.model.fit(x_train, y_train, epochs=epochs, callbacks=[perp_callback], verbose=2)
        
    def buildModelLSTM(self, max_sentences, epochs):
        self.sequence_len = dh.sequence_len
        self.x_train, self.y_train = self.data_handler.build_dataset(max_sentences, self.sequence_len)
        dictionary, reverse_dictionary = self.data_handler.dictionary, self.data_handler.reverse_dict
        vocab_size = len(dictionary)
        
        perp_callback = PrintPerplexity(self, self.dh_val, max_sentences)
        
        self.x_train = x_train.reshape((-1, self.sequence_len)).astype('int32')
        
        self.model = keras.Sequential()
        self.model.add(layers.Embedding(vocab_size+1, emb_dim, mask_zero=True))
        self.model.add(layers.RNN([layers.LSTMCell(self.h_units)], input_shape=(self.sequence_len, self.emb_dim)))
        self.model.add(layers.Dense(vocab_size+1, activation="softmax"))
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy())
        self.model.fit(x_train, y_train, epochs=epochs, callbacks=[perp_callback], verbose=2)
        
# max_sentences = 100
# # words = read_file(filename)
# words = read_lines(filename)
# words = words[:max_sentences]
# words = flatten_sentences(words)
# # dictionary, reverse_dict = build_dictionary(words)
# dictionary, reverse_dict = build_dictionary(words)
# doc_size = len(words)
# vocab_size = len(dictionary) 
# n_hidden = 512
# emb_dim = 10

# rnn_units = n_hidden
# input_dim = emb_dim
# batch_size = 1
# # time_step = 3
# sequence_len = 20





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

# Modified code from
# https://stackoverflow.com/questions/53500047/stop-training-in-keras-when-accuracy-is-already-1-0
class PrintPerplexity(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, curr_mh, dh_val, max_sent):
        super(PrintPerplexity, self).__init__()
        self.dh_val = dh_val
        self.dh_val.build_dictionary(self.dh_val.get_text_list(max_sent))
        self.max_sent =  max_sent
        self.curr_mh = curr_mh
        

    def on_epoch_end(self, epoch, logs=None):
        ph = Perplexity(self.curr_mh, self.model)
        perp, _ = ph.calculate_bigram_perp(self.dh_val, self.max_sent)
        print(f'Epoch {epoch}: Perplexity on validation = {perp}')


trainfile = "penn_train.txt"
testfile = "penn_test.txt"
validfile = "penn_valid.txt"

saved_onlstm = "onlstm_model"
saved_lstm = "lstm_model"

sequence_len_input = 6
sequence_len_output = 1
max_sentences_train = 200
max_sentences_test = 300
emb_dim = 10
n_hidden_on = 100
n_hidden_ls = 110
epochs = 2000
baseline_loss = 3
validation_split = 0.1

dh = DataHandler(trainfile)
dh_test = DataHandler(testfile)
dh_val = DataHandler(validfile)
x_train, y_train = dh.build_dataset(max_sentences_train, sequence_len_input)
mh_ls = ModelHandler(dh, emb_dim, n_hidden_ls, dh_val)
mh_ls.buildModelLSTM(max_sentences_train, epochs)
# mh_ls.model.save(saved_lstm)
ph_ls = Perplexity(mh_ls)
perp_ls, known_ratio_ls = ph_ls.calculate_bigram_perp(dh_test, max_sentences_test)
mh_on = ModelHandler(dh, emb_dim, n_hidden_on, dh_val)
mh_on.buildModelONLSTM(max_sentences_train, epochs)
# mh_on.model.save(saved_onlstm)

ph_on = Perplexity(mh_on)
perp_on, known_ratio_on = ph_on.calculate_bigram_perp(dh_test, max_sentences_test)


# t = tf.constant([1,2.,3]) 
# t =  tf.reshape(t, [batch_size, sequence_len])
# t = t.numpy()
# rng = default_rng()
# x_train, y_train = build_dataset(words)
# # [samples, time_step, feature]
# # x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, padding='post', dtype='int32')
# x_train = x_train.reshape((-1, sequence_len)).astype('int32')
# y_train = y_train.reshape((-1, y_train.shape[0])).T

# y_train = y_train.reshape(-1, y_train.shape[0])
# x_train = x_train.reshape(-1, sequence_len, 1).astype('float32')

# rng = default_rng()
# # initial_state =  [tf.zeros([batch_size, rnn_units]), tf.zeros([batch_size, rnn_units])]
# model = None

# def fit_model():
#     states = [tf.constant(rng.uniform(-1, 1, rnn_units)), tf.constant(rng.uniform(-1, 1, rnn_units))]
#     inputs = keras.Input((sequence_len))
#     # take in [batch_size, input_length] where input_length is the length of a sequence of 
#     # inputs
#     embed = layers.Embedding(vocab_size+1, input_dim, name="embed")
#     # outputs shape of (batch_size, input_length, output_dim)
#     v = embed(inputs)
#     #[batch_size, time_steps, input_dim]
#     onlstm_outputs = layers.RNN(ONLSTM_Cell(rnn_units),name="onlstm")(v)
#     output = layers.Dense(vocab_size+1, activation="softmax", name="output")(onlstm_outputs)
#     model = keras.Model(inputs=inputs, outputs=output)
#     model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy())
#     # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
#     model.fit(x_train, y_train, epochs=200, verbose=2)
    
# fit_model()

# def predict_next(input_words):
#     input_words = np.array([dictionary[word] for word in input_words])
#     input_words = input_words.reshape(-1, len(input_words), 1)
#     res = np.array(model.predict(input_words))
#     res = res.reshape(vocab_size)
#     return reverse_dict[tf.math.argmax(res).numpy()]

# print(predict_next(["the", "mice", "had", "a", "general", "council", "to"]))

# print(model.predict_class(t))

# n_gram = keras.Input(shape=(n_input, ))
# word_emb = layers.Embedding(vocab_size, emb_dim)(n_gram)
# output_gate = layer.Dense()
# y = np.array([1,2,3]).reshape(-1, 1, 3).astype('float32')
# _, state_h, state_c = layers.LSTM(10, return_state=True)(y)
