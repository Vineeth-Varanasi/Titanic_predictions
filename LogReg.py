import numpy as np


class LogisticRegression:

    def __init__(self,lr = 0.1 , n_iters = 2000 , lambda_value = 1):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_value = lambda_value
        self.weights = None
        self.bias = None
        self.costs = []


    def sigmoid(self,Z):
        Z = np.array(Z,dtype=float)
        return 1/(1+np.exp(-Z))
    
    def fit(self , X , Y):

        np_samples , np_features  = X.shape
        self.weights = np.zeros(np_features)
        self.bias = 0
        #starting gradient descent at [0,0,0, ..., 0] , 0
        for _ in range(self.n_iters):
            y_prediction  =  np.dot(X,self.weights) + self.bias
            predictions  = self.sigmoid(y_prediction)

            dw  = (1/np_samples)*np.dot(X.T,  predictions  - Y ) + (self.lambda_value / np_samples)*self.weights
                       #regularized values of w1,w2,w3,w4,w5,...
            db = (1/np_samples)*np.sum(predictions - Y)

            self.weights  = self.weights - self.lr*dw
            self.bias  = self.bias - self.lr*db
            cost = - (1 / np_samples) * np.sum(Y * np.log(predictions ) + (1 - Y) * np.log(1 - predictions ))
            self.costs.append(cost)

        


    def predict(self,X):
        linear_prediction  = np.dot(X,self.weights)+self.bias
        predictions  = self.sigmoid(linear_prediction)  
        return [ 1 if i > 0.5 else 0 for i in predictions]





