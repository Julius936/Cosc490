"""
Dose, H., Møller, J. S., Iversen, H. K., & Puthusserypady, S. (2018). An end-to-end deep learning approach to MI-EEG signal
 classification for BCIs. Expert Systems with Applications, 114, 532–542. https://doi.org/10.1016/j.eswa.2018.08.031
"""

#Addition of the depthwise convolutional layer and the separable convolutional layer to the original ETENet

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D

def create_raw_model(nchan, nclasses, trial_length=80, l1=0):
    """
    CNN model definition
    """
    input_shape = (trial_length, nchan, 1)
    model = Sequential()
    model.add(Conv2D(40, (30, 1), activation="relu", kernel_regularizer=regularizers.l1(l1), padding="same", input_shape=input_shape))
    model.add(DepthwiseConv2D((1, nchan), use_bias = False, depth_multiplier=2))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((30, 1), strides=(15, 1)))
    model.add(SeparableConv2D(80, (16, 1), use_bias = False, padding = 'same'))
    model.add(Flatten())
    model.add(Dense(80, activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    return model
