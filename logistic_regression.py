#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[ ]:


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000, lambda_=1):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.lambda_ = lambda_
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        tol = 1e-5
        prev_loss = 0
        for i in range(self.n_iters):
            y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1/n_samples) * (np.dot(X.T, (y_pred - y))) + ((self.lambda_ / n_samples)*(self.weights))
            db = (1/n_samples) * (np.sum(y_pred - y))
            self.weights = self.weights - (self.lr * dw)
            self.bias = self.bias - (self.lr * db)
            
            current_loss = np.mean(np.square(y_pred - y))
            
            if abs(current_loss - prev_loss) < tol:
                break
                
            prev_loss = current_loss
        
    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)

