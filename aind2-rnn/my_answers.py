from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import numpy as np
# TODO: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    count_loop = (len(series)-window_size)
    X = [series[index:index+window_size] for index in range(count_loop)]
    y = [series[index+window_size] for index in range(count_loop)]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size,1)))
    model.add(Dense(1, activation='linear'))

    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    text = text.lower()

    for c in text:
        if c not in punctuation and (ord(c) >= 97 and  ord(c) <= 122) == False and c != ' ':
            text = text.replace(c,'',1)

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    count_loop = (len(text)-window_size)
    inputs = [text[index:index+window_size] for index in range(0,count_loop,step_size)]
    outputs = [text[index+window_size] for index in range(0,count_loop,step_size)]

    return inputs,outputs

# TODO build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
def build_part2_RNN(window_size, num_chars):

    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size,num_chars)))
    model.add(Dense(num_chars, activation='linear'))
    model.add(Activation('softmax'))

    return model


#window_size = 3
#step_size = 5
#inputs, outputs = window_transform_text("asds askas kaskdm ASAFDSF sdkfmksdmfs d f sd lfsd sdj fkjd sj",window_size,step_size)

#print(inputs)