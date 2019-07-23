import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt # plotting package
import glob

def main():

  labels = [0,1,2,3,4,5,6,7,8,9]

# Read in data
  for i in labels:

      dir = 'LCs'+str(i)+'/'
      lcfiles = glob.glob(dir+'*.dat')

      for file in lcfiles:
          print file
          data = np.loadtxt(file,dtype=float)

# Find locations of zeros, apart from beginning and end
          startind = 0
          startval = data[startind]
          while(startval == 0.0):
             startind += 1
             print startind
             startval = data[startind] 

          endind = len(data) - 1
          endval = data[endind]
          while(endval == 0.0):
             endind -= 1
             endval = data[endind]

          shortdata = data[startind:endind+1]
          indices = np.arange(startind,endind+1)

          zero = np.where(shortdata == 0.0)
          goodpoints = np.where(shortdata != 0.0)
          f = interp1d(goodpoints[0], shortdata[goodpoints])
          shortdata[zero] = f(zero)
          newdata = np.append(data[:startind],shortdata)
          newdata = np.append(newdata,data[endind+1:])


# Plot the result

          plt.clf()
          plt.plot(data, 'o', color='red',markersize=2.5)
          plt.plot(newdata, 'o', color='black',markersize=2.)
          plt.ylabel('Normalized flux')
          plt.savefig(file[0:4]+'EPIC'+file[5:14]+'_interpolated.png')


# Standard boilerplate to call the main() function.
if __name__ == '__main__':


