import numpy as np
from scipy.linalg import lstsq

Data = np.loadtxt('housing.data')

# Slicing
y = Data[:,-1]

print '\nData points', len(y)


# Shortcuts
dot = np.dot
inv = np.linalg.inv

# Dummy features
X = np.ones((506,1))

# Fitting the parameters: theta = (X'*X)^-1*X'*y
# Computing the average of the target value
theta = dot(dot(inv(dot(X.T, X)), X.T), y)
print '\nAverage:', theta


# MSE = (1/N)*sum((y-X*theta)^2)
# Computing MSE obtained using it as a constant prediction
print 'MSE using a constant prediction:', sum((y-dot(X, theta))**2) / len(y)


# Splitting data in 50/50 for training and testing
print '\nSplitting...'
#Training 
half = len(y)/2
H = y[0:half]
print '\nTraining points', len(H)
#print H

#Testing
Z = y[half:len(y)]
print 'Testing points', len(Z)
#print Z

# Training a linear regressor model
print '\nTraining with each variable individually...'

D = np.hstack((X[0:half], Data[0:half,0].reshape(len(H),1)))
theta = lstsq(D,H)[0]
MSE = sum((H-dot(D, theta))**2) / len(H)
print "MSE using variable 0 is: ", MSE

MSE_min = MSE

for number in range(1,len(Data[0])-1):
    D = np.hstack((X[0:half], Data[0:half,number].reshape(len(H),1)))
    theta = lstsq(D,H)[0]
    MSE = sum((H-dot(D, theta))**2) / len(H)
    print 'MSE using variable %d is: ' % (number), MSE
    
    if MSE_min > MSE:
       MSE_min = MSE
       num = number
    
print '\nThe variable %d is the most informative one, getting a MSE of:' % num, MSE_min

# Testing the linear regressor model
print '\nTesting with each variable individually... \n'

D = np.hstack((X[0:half], Data[0:half,0].reshape(len(H),1)))
theta = lstsq(D,H)[0]
MSE = sum((Z-dot(D, theta))**2) / len(Z)
print "MSE using variable 0 is: ", MSE

MSE_min = MSE
MSE_max = MSE

for number in range(1,len(Data[0])-1):
    D = np.hstack((X[0:half], Data[0:half,number].reshape(len(H),1)))
    theta = lstsq(D,H)[0]
    MSE = sum((Z-dot(D, theta))**2) / len(Z)
    print "MSE using variable %d is: " % (number), MSE
    
    if MSE_min > MSE:
        MSE_min = MSE
        num_min = number
        
    if MSE_max < MSE:
        MSE_max = MSE
        num_max = number
        
print '\nThe variable %d is the most generalizable one, getting a MSE of:' % num_min, MSE_min
print '\nThe variable %d is the least generalizable one, getting a MSE of:' % num_max, MSE_max


# Computing the coefficient of determination
# R^2 = 1-FUV

# First, it's needed to have the value of Fraction of Variance Unexplained (FUV)
# FUV = MSE/VAR

# In order to get the variance, it's needed to find the mean
# VAR = (1/N)*sum((mean-y)^2)   mean = (1/N)*sum(y)

print 'Computing the coefficient of determination...'

mean = sum(Z) / len(Z)

VAR = sum((mean-Z)**2) / len(Z)

FUV = MSE_min/VAR

Coeff_Det = 1 - FUV
print '\nThe best coefficient of determination is: ', Coeff_Det

FUV = MSE_max/VAR

Coeff_Det = 1 - FUV
print 'The worst coefficient of determination is: ', Coeff_Det
