import numpy as np
import matplotlib.pyplot as plt # plotting package
import os
import glob

def main():

  labels = [0,1,2,3,4,5,6,7,8,9]

# Initialize light curve data and label 2D arrays:
#  dataarray = np.array([])
#  labelarray = np.array([])

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

#   data_shuf = np.array([])
#   label_shuf = np.array([])
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
   label_test = label_shuf[numtest:]


# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
