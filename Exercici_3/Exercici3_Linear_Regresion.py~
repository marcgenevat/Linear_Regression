import numpy as np
from scipy.linalg import lstsq as theta_calc

data = np.loadtxt('housing.data')

# Slicing
y = data[:,-1]
half = len(y)/2

print '\nData points', len(y)


# Shortcuts
dot = np.dot
inv = np.linalg.inv

# Dummy features
X = np.ones((506,1))

# Fitting the parameters: theta = (X'*X)^-1*X'*y
# Computing the average of the target value
theta = dot(dot(inv(dot(X.T, X)), X.T), y)

# MSE = (1/N)*sum((y-X*theta)^2)
# Computing MSE obtained using it as a constant prediction
print sum((y-dot(X, theta))**2) / len(y)


# Splitting data in 50/50 for training and testing
print 'Splitting...'

#Training 
data_train = data[:half,:-1]
#print 'Training points', len(data_train)

input_train = data[:half,-1]

#Testing
data_test = data[half:,:-1]
#print 'Testing points', len(data_test)

input_test = data[half:,-1]

# Training a linear regressor model
print '\nTraining with each variable individually...'

list_mse=[]
for i in range(0,len(data[0])-1):
    variables = np.hstack((X[:half], data_train[:,i].reshape(half,1)))
    theta = theta_calc(variables,input_train)[0]
    #list_mse.append(calcular_mse(X,data_train[:,i]))
    list_mse.append(sum((input_train-dot(variables, theta))**2) / half)

print '\nThe variable %d is the most informative one, getting a MSE of:' % list_mse.index(min(list_mse)), min(list_mse)

# Testing the linear regressor model
print '\nTesting with each variable individually... \n'

list_mse=[]
for i in range(0,len(data[0])-1):
    variables = np.hstack((X[:half], data_test[:,i].reshape(half,1)))
    list_mse.append(sum((input_test-dot(variables, theta))**2) / half)
    
print '\nThe most generalizable variable is:', list_mse.index(min(list_mse)), 'getting a MSE of:', min(list_mse)
print '\nThe worst generalizable variable is:', list_mse.index(max(list_mse)), 'getting a MSE of:', max(list_mse)


# Computing the coefficient of determination
# R^2 = 1-FUV

# First, it's needed to have the value of Fraction of Variance Unexplained (FUV)
# FUV = MSE/VAR

# In order to get the variance, it's needed to find the mean
# VAR = (1/N)*sum((mean-y)^2)   mean = (1/N)*sum(y)

mean = sum(input_test) / half

VAR = sum((mean-input_test)**2) / half

FUV = list_mse/VAR

Coeff_Det = 1 - FUV

print 'The most useful variable is %d giving a coefficient of determination of:' % Coeff_Det.argmax(axis=0), max(Coeff_Det)

