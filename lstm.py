"""
@Article{gumpy2018,
    Title = {gumpy: A Python Toolbox Suitable for Hybrid Brain-Computer Interfaces},
    Author = {Tayeb, Zied and Waniek, Nicolai and Fedjaev, Juri and Ghaboosi, Nejla and Rychly, Leonard and Widderich, Christian and Richter, Christoph and Braun, Jonas and Saveriano, Matteo and Cheng, Gordon and Conradt, Jorg},
    Year = {2018},
    Journal = {}
}
"""
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM as _LSTM
import tensorflow as tf

class LSTM(tf.keras.Model):

    def create_model(input_shape, num_hidden_neurons=128,
                      num_layers=1, dropout=0.2, recurrent_dropout=0.2,
                      print_summary=False):

        model = Sequential()
        if num_layers > 1:
            for i in range(1, num_layers, 1):
                model.add(_LSTM(num_hidden_neurons, input_shape=input_shape,
                               return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
            model.add(_LSTM(num_hidden_neurons))
        else:
            model.add(_LSTM(num_hidden_neurons, input_shape=input_shape, dropout=dropout,
                           recurrent_dropout=recurrent_dropout))
        model.add(Dense(2, activation='softmax'))

        if print_summary:
            print(model.summary())

        # compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # assign and return
        #self.model = model
        return model

