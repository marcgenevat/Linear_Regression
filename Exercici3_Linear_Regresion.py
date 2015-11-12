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
Bias = np.ones((len(y),1))	#Bias term vector

# Fitting the parameters: theta = (X'*X)^-1*X'*y
# Computing the average of the target value
theta = theta_calc(Bias,y)[0]

print 'The average of the target value is:' , theta

# MSE = (1/N)*sum((y-X*theta)^2)
# Computing MSE obtained using it as a constant prediction
print '\nMSE as a constant prediction: ' , MSE(y,Bias,theta)


print '\n\n#####Solving Q3#####'

# Splitting data in 50/50 for training and testing

#Training 
data_train = data[:half,:-1]	#Training features
output_train = data[:half,-1]	#Training labels

#Testing
data_test = data[half:,:-1]		#Testing features
output_test = data[half:,-1]	#Testing labels

# Training and testing a linear regressor model
print '\nTraining and testing with each variable individually...'

list_mse_train=[]
list_theta=[]
list_mse_test=[]
for i in range(0,len(data[0])-1):
	train_features = np.hstack((Bias[:half], data_train[:,i].reshape(half,1)))	#Stack a feature array (vector) with Bias term (training)
	test_features = np.hstack((Bias[:half], data_test[:,i].reshape(half,1)))	#Stack a feature array (vector) with Bias term (testing)
	list_theta.append(theta_calc(train_features,output_train)[0])			#Save every theta for each different feature	
	list_mse_train.append(MSE(output_train,train_features,list_theta[i]))		#Save every MSE for each theta to a list (training)
	list_mse_test.append(MSE(output_test,test_features,list_theta[i]))	#Save every MSE for each theta to a list (testing)

# Answering the Q3's questions
index = list_mse_train.index(min(list_mse_train)) + 1
print '\nThe variable', index, 'is the most informative one, getting a MSE of:', min(list_mse_train)
print '\nThe Average number of rooms per dwelling (X6) is a possible measure for the size of the houses, \nso it is expeted to be strongly correlated with the price of the houses'

index = list_mse_test.index(min(list_mse_test)) + 1
print '\n\nThe most generalizable variable is the number', index, 'getting a MSE of:', min(list_mse_test)
print '\nThe Proportion of lower status of the population (X13) exhibits the clearest negative relation with the price of the houses, \nso the price would directly depend of the status of the population at the district'

index = list_mse_test.index(max(list_mse_test)) + 1
print '\n\nThe worst generalizable variable is the number', index, 'getting a MSE of:', max(list_mse_test)
print '\nThe Per-capita crime rate (X1) presents a very bad respond to the model \ndue to there could be different group of houses with a similar rate but very different price'

# Computing the coefficient of determination
mean = sum(output_test) / half		#mean = (1/N)*sum(y)
VAR = sum((mean-output_test)**2) / half		#VAR = (1/N)*sum((mean-y)^2)
FUV = list_mse_test/VAR		#Fraction of Variance Unexplained (FUV) -> FUV = MSE/VAR
Coeff_Det_Q3 = 1 - FUV		#Coefficient of determination -> R^2 = 1-FUV

index = Coeff_Det_Q3.argmax(axis=0)
print '\n\nThe most useful variable is', index,'giving a coefficient of determination (R^2) of: ', max(Coeff_Det_Q3)
print '\nIndeed, the most useful variable is the Proportion of lower status of the population (X13)'


print '\n\n#####Solving Q4#####'

print '\nTraining and testing a model with all the variables plus a bias term...'

# Taking a training input with all the variables plus the bias term
train_features = np.hstack((Bias[:half], data_train.reshape(half,data_train.shape[1]))) #Stack all features with Bias term (training)
test_features = np.hstack((Bias[:half], data_test.reshape(half,data_test.shape[1]))) #Stack all features with Bias term (testing)
theta = theta_calc(train_features,output_train)[0]	#Calculating the model (theta)
MSE_train = MSE(output_train,train_features,theta)	#Calculating the error (training)
MSE_test = MSE(output_test,test_features,theta)			#Calculating the error (testing)

mean = sum(output_test) / half		#mean = (1/N)*sum(y)
VAR = sum((mean-output_test)**2) / half		#VAR = (1/N)*sum((mean-y)^2)
FUV = MSE_test / VAR		#Fraction of Variance Unexplained (FUV) -> FUV = MSE/VAR
Coeff_Det_Q4 = 1 - FUV		#Coefficient of determination -> R^2 = 1-FUV

print '\nThe resulting MSE in the training phase is:', MSE_train
print '\nThe resulting MSE in the testing phase is:', MSE_test
print '\nThe resulting Coefficient of determination is:', Coeff_Det_Q4


print '\n\nTrying over without the worst-performing variable found at the Q3...'

# Removing the worst-performing variable in the features array
data_train_Q4 = np.delete(data_train,Coeff_Det_Q3.argmin(axis=0),axis=1)
data_test_Q4 = np.delete(data_test,Coeff_Det_Q3.argmin(axis=0),axis=1)

# Starting over the experiment
train_features = np.hstack((Bias[:half], data_train_Q4.reshape(half,data_train_Q4.shape[1])))	#Stack all features with Bias term (training)
test_features = np.hstack((Bias[:half], data_test_Q4.reshape(half,data_test_Q4.shape[1]))) #Stack all features with Bias term (testing)
theta = theta_calc(train_features,output_train)[0]	#Calculating the model (theta)
MSE_train = MSE(output_train,train_features,theta)	#Calculating the error (training)
MSE_test = MSE(output_test,test_features,theta)			#Calculating the error (testing)

mean = sum(output_test) / half		#mean = (1/N)*sum(y)
VAR = sum((mean-output_test)**2) / half		#VAR = (1/N)*sum((mean-y)^2)
FUV = MSE_test / VAR		#Fraction of Variance Unexplained (FUV) -> FUV = MSE/VAR
Coeff_Det_Q4 = 1 - FUV		#Coefficient of determination -> R^2 = 1-FUV

print '\nThe resulting MSE in the training phase is:' , MSE_train
print '\nThe resulting MSE in the testing phase is:' , MSE_test
print '\nThe resulting Coefficient of determination is:' , Coeff_Det_Q4

print '\nThe model is much better than before, so there were variables which were effecting badly at the model'


print '\n\n#####Solving Q5#####'

print '\nTraining and testing a model with all the variables plus a bias term applying non-linear transformations to them...'

list_mse_train=[]
list_theta=[]
list_mse_test=[]

for i in range(0,len(data[0])-1):
    x1_train = data_train[:,i].reshape(half,1)        #X^1
    x2_train = data_train[:,i].reshape(half,1) ** 2   #X^2
    x3_train = data_train[:,i].reshape(half,1) ** 3   #X^3
    x4_train = data_train[:,i].reshape(half,1) ** 4   #X^4
    train_features = np.hstack((Bias[:half], x1_train, x2_train, x3_train, x4_train))   #(1 + X^1 + X^2 + X^3 + X^4)
    
    list_theta.append(theta_calc(train_features,output_train)[0])   #Calculating the model (theta)
    
    x1_test = data_test[:,i].reshape(half,1)					#X^1
    x2_test = data_test[:,i].reshape(half,1) ** 2     #X^2
    x3_test = data_test[:,i].reshape(half,1) ** 3			#X^3
    x4_test = data_test[:,i].reshape(half,1) ** 4			#X^4
    test_features = np.hstack((Bias[:half], x1_test, x2_test, x3_test, x4_test))   #(1 + X^1 + X^2 + X^3 + X^4)
    
    list_mse_train.append(MSE(output_train,train_features,list_theta[i]))   #Calculating the error (training)
    list_mse_test.append(MSE(output_test,test_features,list_theta[i]))      #Calculating the error (testing)
    
    mean = sum(output_test) / half		#mean = (1/N)*sum(y)
    VAR = sum((mean-output_test)**2) / half		#VAR = (1/N)*sum((mean-y)^2)
    FUV = list_mse_test / VAR		#Fraction of Variance Unexplained (FUV) -> FUV = MSE/VAR
    Coeff_Det_Q5 = 1 - FUV		#Coefficient of determination -> R^2 = 1-FUV
    
# Answering the Q5's questions
index = list_mse_train.index(min(list_mse_train)) + 1
print '\nThe variable', index,'is the most informative one, getting a MSE of:', min(list_mse_train)
print 'The Average number of rooms (X6) per dwelling still being the most informative one with even a lower MSE'

index = list_mse_test.index(min(list_mse_test)) + 1
print '\nThe most generalizable variable is', index, 'getting a MSE of:', min(list_mse_test)
print 'The Proportion of lower status of the population (X13) still being the most generalizable one with even a lower MSE'

index = list_mse_test.index(max(list_mse_test)) + 1
print '\nThe worst generalizable variable is', index, 'getting a MSE of:', max(list_mse_test)
print 'The Per-capita crime rate (X1) still being the worst generalizable one with even a much higher MSE'

index = Coeff_Det_Q5.argmax(axis=0) + 1
print '\nThe best case is for the variable', index,'getting a R^2 of:', max(Coeff_Det_Q5)
print 'So the best fitted model (got with the X13) has improved a little bit!'
print 'But as we can see, this metode does a horrible work depending of the variable.'



print '\n\n#####Solving Q6#####'


def reg_lin_regr(theta,features,labels, Lambda):
    
    f = (((dot(features, sum(theta[1:]) ** 2) - labels) ** 2) / len(y)) + Lambda * sum(theta[1:] ** 2)   # f = (1/N) * (X*theta - y)^2 + landa*theta
    
    return f

def derv_reg_lin_regr(theta, features, labels, Lambda):
    
    f_prima = ((2 * (features.transpose() * (dot(features, theta) - labels))) / len(labels)) + 2 * Lambda * theta
    # f' = (2/N) * ((X^T)*(X*theta-y)) + 2*landa*theta
    
    return f_prima

def gradient_descent():

    theta = [None for x in range(maxit)]
    loss = []
    theta = np.random.sample()
    
    step = 1e-6
    maxit = len(y); loss = zeros(maxit)
    
    loss = reg_lin_regr(theta)
    for t in range(1, maxit):
        theta[t] = theta[t-1] - step * derv_reg_lin_regr(theta[t-1])
        loss[t] = reg_lin_regr(theta[t])
        
    return theta[maxit]
