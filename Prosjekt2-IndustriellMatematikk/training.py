import numpy as np
from layers import *
from neural_network import NeuralNetwork
from utils import onehot

def algorithm_4(x, y, n_iter, alpha, m, neuralnet):
    n_samples = x.shape[0]
    batch_size = y.shape[1]
    n_batches = n_samples // batch_size
    
    EpochLosses = []  # To store average loss per epoch
    for epoch in range(n_iter):
        # Shuffle data at the beginning of each epoch
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        x_shuffled = x[indices]
        y_shuffled = y[indices]
        
        total_loss = 0  # Accumulate loss over batches for the current epoch
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = onehot(x_shuffled[start_idx:end_idx], m)
            Y_batch = y_shuffled[start_idx:end_idx]
            
            Z = neuralnet.forward(X_batch)
            batch_loss = neuralnet.loss.forward(Z, Y_batch)
            total_loss += batch_loss  # Accumulate loss
            
            dLdz = neuralnet.loss.backward()
            neuralnet.backward(dLdz)
            neuralnet.step_adam(epoch + 1, alpha)  # epoch + 1 to avoid division by zero in Adam
        
        avg_loss = total_loss / n_batches  # Calculate average loss for the epoch
        EpochLosses.append(avg_loss)  # Store the average loss
        
        print(f'Epoch {epoch+1}/{n_iter}, Average Loss: {avg_loss}')  # Print average loss for the epoch
    
    return EpochLosses

def algorithm_4_sort(x, y, n_iter, alpha, m, neuralnet):
    batch_size, n_samples, _ = x.shape
    
    EpochLosses = []  # To store average loss per epoch
    for epoch in range(1,n_iter):
        # Shuffle data at the beginning of each epoch
        total_loss = []
        for k in range(batch_size):
            X_batch = onehot(x[k], m)
            Z = neuralnet.forward(X_batch)
            batch_loss = neuralnet.loss.forward(Z, y[k][:,-5:])
            total_loss.append(batch_loss)
            dLdz = neuralnet.loss.backward()
            neuralnet.backward(dLdz)
            neuralnet.step_adam(n_iter, alpha)  # epoch + 1 to avoid division by zero in Adam
            print(f'loss for batch {epoch} is {batch_loss}')
        
        avg_loss = np.mean(total_loss)  # Calculate average loss for the epoch
        EpochLosses.append(avg_loss)  # Store the average loss
        
        print(f'Epoch {epoch}/{n_iter}, Average Loss: {avg_loss}')  # Print average loss for the epoch
    
    return EpochLosses

def sorting(neuralnet, x, y,m):
    X = onehot(x, m)
    Z = neuralnet.forward(X)
    z_hat = np.argmax(Z, axis = 1)
    amount = 0
    for i in range(y.shape[0]):
        y[i] = y[i].astype(int)
        if np.array_equal(y[i], z_hat[i][::-1]):
            amount += 1
    return amount/y.shape[0]
