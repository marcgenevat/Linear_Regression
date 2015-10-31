import numpy as np

Data = np.loadtxt('housing.data')

#slicing
y = Data[:,-1]

print 'Data points', len(y)

print y.T
