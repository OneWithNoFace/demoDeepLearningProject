import os 
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Activation
from keras.models import Sequential

class CNN1:
  @staticmethod
  def build(num_channels,img_rows,img_cols,num_classes,activation='relu'):

    input_shape = (img_rows,img_cols,num_channels)

    model = Sequential()
    model.add(Conv2D(20,(5,5),input_shape=input_shape))
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Conv2D(50,(3,3)))
    model.add(Activation(activation))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation(activation))

    model.add(Dense(9))
    model.add(Activation('softmax'))

    return model