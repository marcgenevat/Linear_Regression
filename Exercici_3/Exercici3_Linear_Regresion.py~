import numpy as np
from scipy.linalg import lstsq as theta_calc

data = np.loadtxt('housing.data')

# MSE Function
def MSE(y,X,theta):	
	MSE = float(sum((y-dot(X, theta))**2) / len(y))
	return MSE
	

# Slicing
y = data[:,-1]  #Outputs (labels)
half = len(y)/2

print '\nHave a dataset of %d data points' % len(y)

# Shortcuts
dot = np.dot	#Multiplier
inv = np.linalg.inv #Inversor

# Dummy features
Bias = np.ones((506,1))	#Bias term vector

# Fitting the parameters: theta = (X'*X)^-1*X'*y
# Computing the average of the target value
theta = dot(dot(inv(dot(Bias.T, Bias)), Bias.T), y)

# MSE = (1/N)*sum((y-X*theta)^2)
# Computing MSE obtained using it as a constant prediction
print '\nMSE as a constant prediction: ' , MSE(y,Bias,theta)


# Splitting data in 50/50 for training and testing
print '\nSplitting data...'

#Training 
data_train = data[:half,:-1]	#Training features

output_train = data[:half,-1]	#Training labels

#Testing
data_test = data[half:,:-1]		#Testing features

output_test = data[half:,-1]	#Testing labels

# Training a linear regressor model
print '\n\nTraining with each variable individually...'

list_mse_train=[]
list_theta=[]
for i in range(0,len(data[0])-1):
    train_features = np.hstack((Bias[:half], data_train[:,i].reshape(half,1)))	#Stack feature array (vector) with Bias term
    list_theta.append(theta_calc(train_features,output_train)[0])			#Save every theta for each different feature	
    list_mse_train.append(MSE(output_train,train_features,list_theta[i]))		#Save every MSE for each theta to a list

print '\nThe variable %d is the most informative one, getting a MSE of:' % list_mse_train.index(min(list_mse_train)), min(list_mse_train)

# Testing the linear regressor model
print '\n\nTesting with each variable individually...'

list_mse_test=[]
for i in range(0,len(data[0])-1):
    test_features = np.hstack((Bias[:half], data_test[:,i].reshape(half,1)))	#Stack feature array (vector) with Bias term
    list_mse_test.append(MSE(output_test,test_features,list_theta[i]))	#Save every MSE for each theta to a list
    
print '\nThe most generalizable variable is:', list_mse_test.index(min(list_mse_test)), 'getting a MSE of:', min(list_mse_test)
print '\nThe worst generalizable variable is:', list_mse_test.index(max(list_mse_test)), 'getting a MSE of:', max(list_mse_test)


# Computing the coefficient of determination
# R^2 = 1-FUV

# First, it's needed to have the value of Fraction of Variance Unexplained (FUV)
# FUV = MSE/VAR

# In order to get the variance, it's needed to find the mean
# VAR = (1/N)*sum((mean-y)^2)   mean = (1/N)*sum(y)

mean = sum(output_test) / half

VAR = sum((mean-output_test)**2) / half

FUV = list_mse_test/VAR

Coeff_Det = 1 - FUV

print '\n\nThe most useful variable is %d giving a coefficient of determination (R^2) of: ' % Coeff_Det.argmax(axis=0), max(Coeff_Det)
print '\n'
