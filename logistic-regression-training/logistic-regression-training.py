import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    b = 0.0;
    w = np.zeros(n_features)
    
    for _ in range(steps):
        z = np.dot(X,w) + b
        p = _sigmoid(z);
        e = p-y
        dldw = np.dot(X.T,e)
        dldw = dldw/n_samples
        dldb = np.mean(e)
        w = w - lr * dldw
        b = b - lr * dldb
    return w,b