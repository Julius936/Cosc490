#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author: Bruce Shuyue Jia
@Date: Jan 30, 2021
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from prepare_data import prepare_data
from tensorflow.keras.callbacks import EarlyStopping

datasets = input("ENTER THE 2 DATASETS: ")
separate_datasets = datasets.split('&')
A=separate_datasets[0]
B=separate_datasets[1]

category_1 = "/home/hzy/synthdata/ica/class19"+A
category_2 = "/home/hzy/synthdata/raw"


csv_file_path = "/home/julius/Transformer/csv_files/"
num_channels = 19
num_samples = 511

prepare_data(category_1, category_2)

# Read Training Data
train_data = pd.read_csv(csv_file_path + 'training_set.csv', header=None)
train_data = np.array(train_data).astype('float32')

# Read Training Labels
train_labels = pd.read_csv(csv_file_path + 'training_label.csv', header=None)
train_labels = np.array(train_labels).astype('float32')
train_labels = np.squeeze(train_labels)

# Read Testing Data
test_data = pd.read_csv(csv_file_path + 'test_set.csv', header=None)
test_data = np.array(test_data).astype('float32')

# Read Testing Labels
test_labels = pd.read_csv(csv_file_path + 'test_label.csv', header=None)
test_labels = np.array(test_labels).astype('float32')
test_labels = np.squeeze(test_labels)

# Read validation Data
validation_data = pd.read_csv(csv_file_path + 'validation_set.csv', header=None)
validation_data = np.array(validation_data).astype('float32')

# Read validation Labels
validation_labels = pd.read_csv(csv_file_path + 'validation_label.csv', header=None)
validation_labels = np.array(validation_labels).astype('float32')
validation_labels = np.squeeze(validation_labels)

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.5):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out = self.layernorm2(out1 + ffn_output)
        return out


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=maxlen, delta=1) 
        positions = self.pos_emb(positions)
        x = tf.reshape(x, [-1, maxlen, embed_dim])
        out = x + positions
        return out

maxlen = 19     
embed_dim = 511 
num_heads = 8   # Number of attention heads
ff_dim = 64     # Hidden layer size in feed forward network inside transformer

# Input Time-series
inputs = layers.Input(shape=(maxlen*embed_dim,))
embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
x = embedding_layer(inputs)

# Encoder Architecture
transformer_block_1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
transformer_block_2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
x = transformer_block_1(x)
x = transformer_block_2(x)

# Output
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="binary_crossentropy", metrics = ['accuracy'])
              #metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall()])
early_stopping = EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')

history = model.fit(
    train_data, train_labels, batch_size=128, epochs=1000, validation_data=(validation_data, validation_labels), callbacks=[early_stopping]
)

#model.save('MyModel_tf',save_format='tf')

test_evalauation = model.evaluate(test_data, test_labels, verbose=1)

print('Test loss:', test_evalauation[0])
print('Test accuracy:', test_evalauation[1])
