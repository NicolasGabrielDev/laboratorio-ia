import numpy as np
import json
import os

class MLP_TDNN:
    def __init__(self, p, N1, lr=0.1, momentum=0.8, precision=0.5e-6, max_epochs=100000, seed=None):
        self.p = p
        self.N1 = N1
        self.lr = lr
        self.momentum = momentum
        self.precision = precision
        self.max_epochs = max_epochs
        
        if seed is not None:
            np.random.seed(seed)
            
        # Weights initialization between 0 and 1
        # W1: inputs to hidden layer, size: (p, N1)
        # B1: hidden layer bias, size: (1, N1)
        self.W1 = np.random.rand(p, N1)
        self.B1 = np.random.rand(1, N1)
        
        # W2: hidden to output layer, size: (N1, 1)
        # B2: output layer bias, size: (1, 1)
        self.W2 = np.random.rand(N1, 1)
        self.B2 = np.random.rand(1, 1)
        
        # Momentum components
        self.v_W1 = np.zeros_like(self.W1)
        self.v_B1 = np.zeros_like(self.B1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_B2 = np.zeros_like(self.B2)
        
    def sigmoid(self, x):
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
        
    def sigmoid_deriv(self, out):
        return out * (1 - out)
        
    def forward(self, X):
        # Hidden layer
        self.Z1 = np.dot(X, self.W1) + self.B1
        self.A1 = self.sigmoid(self.Z1)
        
        # Output layer
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2
        
    def train(self, X, y):
        N = X.shape[0]
        eqm_history = []
        
        for epoch in range(self.max_epochs):
            # Forward pass
            out = self.forward(X)
            
            # Error and EQM
            error = y - out
            eqm = np.mean(error ** 2)
            eqm_history.append(eqm)
            
            # Stopping criterion
            if epoch > 0 and abs(eqm_history[-1] - eqm_history[-2]) <= self.precision:
                break
                
            # Backpropagation
            # Output layer error gradient
            d_out = error * self.sigmoid_deriv(out)
            
            # Hidden layer error gradient
            d_hidden = np.dot(d_out, self.W2.T) * self.sigmoid_deriv(self.A1)
            
            # Gradients
            grad_W2 = np.dot(self.A1.T, d_out) / N
            grad_B2 = np.sum(d_out, axis=0, keepdims=True) / N
            
            grad_W1 = np.dot(X.T, d_hidden) / N
            grad_B1 = np.sum(d_hidden, axis=0, keepdims=True) / N
            
            # Update with momentum
            self.v_W2 = self.momentum * self.v_W2 + self.lr * grad_W2
            self.v_B2 = self.momentum * self.v_B2 + self.lr * grad_B2
            self.v_W1 = self.momentum * self.v_W1 + self.lr * grad_W1
            self.v_B1 = self.momentum * self.v_B1 + self.lr * grad_B1
            
            self.W2 += self.v_W2
            self.B2 += self.v_B2
            self.W1 += self.v_W1
            self.B1 += self.v_B1
            
        return eqm_history, epoch + 1
        
    def predict(self, X):
        return self.forward(X)

def prepare_data(data_dict, p):
    """
    Prepares Time Delay NN dataset.
    data_dict: dictionary with keys 't' and values 'f(t)'
    p: number of delays
    Returns X, y
    """
    times = sorted([int(k) for k in data_dict.keys()])
    values = [data_dict[str(t)] for t in times]
    
    X = []
    y = []
    for i in range(len(values) - p):
        # delays: t-1, t-2, ..., t-p -> reversed: [values[i+p-1], values[i+p-2], ..., values[i]]
        # The prompt says x1=x(t-1), x2=x(t-2)... So order is [x(t-1), x(t-2), ..., x(t-p)]
        # i.e., values from t-1 down to t-p
        window = values[i : i+p][::-1] 
        X.append(window)
        y.append(values[i+p])
        
    return np.array(X), np.array(y).reshape(-1, 1)

def get_test_features(train_dict, test_dict, p):
    """
    Prepares test dataset.
    For test samples, t=101 to 120, we need the past 'p' values.
    Some of these values might come from the training dataset.
    """
    # Combine data to get historical values
    combined = {**train_dict, **test_dict}
    times = sorted([int(k) for k in test_dict.keys()])
    
    X = []
    y = []
    for t in times:
        window = []
        for delay in range(1, p + 1):
            window.append(combined[str(t - delay)])
        X.append(window)
        y.append(test_dict[str(t)])
        
    return np.array(X), np.array(y).reshape(-1, 1)

