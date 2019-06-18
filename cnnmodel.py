import numpy as np
import matplotlib.pyplot as plt # plotting package
import os
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def main():

  labels = [0,1,2,3,4,5,6,7,8,9]


# Read in data
  for i in labels:
      labellist = np.zeros(10,dtype=int)
      labellist[i] = 1

      dir = 'LCs'+str(labels[i])
      lcfiles = glob.glob(dir+'/*.dat')

      for file in lcfiles:
          data = np.loadtxt(file,dtype=float)
          if ((file == lcfiles[0]) & (i == 0)):
            dataarray = data
            labelarray = labellist
          else:
            dataarray = np.vstack((dataarray,data))
            labelarray = np.vstack((labelarray,labellist))

# Now shuffle it.

   index_shuf = np.arange(len(labelarray))
   np.random.shuffle(index_shuf)
   for i in index_shuf:
     if (i == index_shuf[0]):
       data_shuf = dataarray[i]
       label_shuf = labelarray[i]
     else:
       data_shuf = np.vstack((data_shuf,dataarray[i]))
       label_shuf = np.vstack((label_shuf,labelarray[i]))


# Chop off 25% of the data for testing; leave 75% for training:

   numdata = len(label_shuf)
   numtest = int(numdata/4)
   data_test = data_shuf[:numtest]
   label_test = label_shuf[:numtest]

   data_train = data_shuf[numtest:]
   label_train = label_shuf[numtest:]


# Define model
  n_features = 1
  n_steps = numdata - numtest

  model = Sequential()
  model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
  model.add(MaxPooling1D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(50, activation='relu'))
  model.add(Dense(1))
  model.compile(optimizer='adam', loss='mse')

# Fit model
  model.fit(data_train, label_train, epochs=10, verbose=0)

# demonstrate prediction
x_input = data_test[0]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
