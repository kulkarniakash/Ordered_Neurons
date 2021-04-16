# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 00:18:04 2021

@author: kulka
"""
import numpy as np
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
emb_dim = 10
n_input = 7

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
    
def convert_to_word(one_hot):
    ind = tf.math.argmax(one_hot)
    return reverse_dict[ind.numpy()]
    


model = keras.Sequential()
model.add(layers.Embedding(vocab_size, emb_dim))
model.add(layers.RNN([layers.LSTMCell(n_hidden, recurrent_regularizer='l2'), layers.LSTMCell(n_hidden, recurrent_regularizer='l2')], input_shape=(n_input, emb_dim)))
model.add(layers.Dense(vocab_size, activation="softmax"))
model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy())
x_train, y_train = build_dataset(words)
# y_train = y_train.reshape(-1, y_train.shape[0])
x_train = x_train.reshape(-1, n_input, 1).astype('float32')
# tf.reshape(x_train, shape=(x_train.get_shape()[0], x_train.get_shape()[1], x_train.get_shape()[2]))
#  = tf.reshape(y_train, shape=(-1, y_train))
model.fit(x_train, y_train, epochs=100, verbose=2)


def predict_next(input_words):
    input_words = np.array([dictionary[word] for word in input_words])
    input_words = input_words.reshape(-1, n_input, 1)
    res = np.array(model.predict(input_words))
    res = res.reshape(vocab_size)
    return reverse_dict[tf.math.argmax(res).numpy()]

def loop(words, times):
    seq = words
    for i in range(times):
        new_word = predict_next(seq)
        print(new_word, end=' ')
        seq = seq[1:] + [new_word]
predict_next(["general", "council", "to"])