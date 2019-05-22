import numpy as np
import matplotlib.pyplot as plt # plotting package
import os

def main():

  labels = [0,1,2,3,4,5,6,7,8,9]

# Initialize light curve data and label 2D arrays:
  dataarray = np.array([])
  labelarray = np.array([])

# Read in data
  for i in labels:
      labellist = np.zeros(10,dtype=int)
      labellist[varnum] = 1

      dir = 'LCs'+str(labels)
      lcfiles = os.listdir(dir)

       for file in lcfiles:
          data = np.loadtxt('file',dtype=float)
          dataarray = np.vstack((dataarray,data))
          labelarray = np.vstack((labelarray,labellist))


# Standard boilerplate to call the main() function.
if __name__ == '__main__':
  main()
