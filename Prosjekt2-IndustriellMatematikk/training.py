import numpy as np
from layers import *
from neural_network import NeuralNetwork
from utils import onehot


def algorithm_4_sort_finished(x, y, n_iter, alpha, m, neuralnet, r):
    n_batches = x.shape[0]  # Number of batches in the dataset

    EpochLosses = []  # To store average loss per epoch

    for epoch in range(n_iter):
        total_loss = 0  # Accumulate loss over all batches for the current epoch

        for i in range(n_batches):
            X_batch = onehot(x[i], m)  # Onehot encode the current batch of x
            Y_batch = y[i]  # Corresponding y batch

            Z = neuralnet.forward(X_batch)
            # Adjust the slicing of Z to only include the predictions for the sorted sequence
            # Assuming the last 'r' elements of Z correspond to the sorted sequence
            batch_loss = neuralnet.loss.forward(Z, Y_batch[:, -r:])
            total_loss += batch_loss  # Accumulate loss

            dLdz = neuralnet.loss.backward()  # Compute gradients for the loss
            neuralnet.backward(dLdz)  # Backpropagate through the network
            neuralnet.step_adam(epoch + 1, alpha)  # Apply Adam optimizer

        avg_loss = total_loss / n_batches  # Calculate average loss for the epoch
        EpochLosses.append(avg_loss)  # Store the average loss

        if (epoch + 1) % 10 == 0:  # Condition to print only every 10th epoch
            print(f"Epoch {epoch+1}/{n_iter}, Average Loss: {avg_loss}")

    return EpochLosses


def algorithm_4_addition_finished(x, y, n_iter, alpha, m, neuralnet, r):
    n_batches = x.shape[0]  # Number of batches in the dataset

    EpochLosses = []  # To store average loss per epoch

    for epoch in range(n_iter):
        total_loss = 0  # Accumulate loss over all batches for the current epoch

        for i in range(n_batches):
            X_batch = onehot(x[i], m)  # Onehot encode the current batch of x
            Y_batch = y[i]  # Corresponding y batch

            Z = neuralnet.forward(X_batch)
            # Adjust the slicing of Z to only include the predictions for the sorted sequence
            # Assuming the last 'r' elements of Z correspond to the sorted sequence
            batch_loss = neuralnet.loss.forward(Z, Y_batch[:, -(r + 1) :])
            total_loss += batch_loss  # Accumulate loss

            dLdz = neuralnet.loss.backward()  # Compute gradients for the loss
            neuralnet.backward(dLdz)  # Backpropagate through the network
            neuralnet.step_adam(epoch + 1, alpha)  # Apply Adam optimizer

        avg_loss = total_loss / n_batches  # Calculate average loss for the epoch
        EpochLosses.append(avg_loss)  # Store the average loss

        if (epoch + 1) % 10 == 0:  # Condition to print only every 10th epoch
            print(f"Epoch {epoch+1}/{n_iter}, Average Loss: {avg_loss}")

    return EpochLosses


def algorithm_4_sort(x, y, n_iter, alpha, m, neuralnet, r):
    batch_size, n_samples, _ = x.shape

    EpochLosses = []  # To store average loss per epoch
    for epoch in range(1, n_iter+1):
        # Shuffle data at the beginning of each epoch
        total_loss = []
        for k in range(batch_size):
            X_batch = onehot(x[k], m)
            Z = neuralnet.forward(X_batch)
            batch_loss = neuralnet.loss.forward(Z, y[k][:, -r:])
            total_loss.append(batch_loss)
            dLdz = neuralnet.loss.backward()
            neuralnet.backward(dLdz)
            neuralnet.step_adam( n_iter, alpha
            )  # epoch + 1 to avoid division by zero in Adam
            # print(f"loss for batch {epoch} is {batch_loss}")

        avg_loss = np.mean(total_loss)  # Calculate average loss for the epoch
        EpochLosses.append(avg_loss)  # Store the average loss

        print(
            f"Epoch {epoch}/{n_iter}, Average Loss: {avg_loss}"
        )  # Print average loss for the epoch

    return EpochLosses


def algorithm_4_add(x, y, n_iter, alpha, m, r, neuralnet):
    batch_size, n_samples, _ = x.shape

    loss = []  # liste for å plotte senere
    for i in range(1, n_iter):
        total_loss = []
        for k in range(batch_size):
            X_batch = onehot(x[k], m)
            Z = neuralnet.forward(X_batch)
            batch_loss = neuralnet.loss.forward(Z, y[k][:, -(r + 1) :])
            total_loss.append(batch_loss)
            dLdz = neuralnet.loss.backward()
            neuralnet.backward(dLdz)
            neuralnet.step_adam(
                n_iter, alpha
            )  # epoch + 1 to avoid division by zero in Adam
            # print(f'loss for batch {k} and iteration {i} is {batch_loss}')

        avg_loss = np.mean(total_loss)  # Calculate average loss for the epoch
        loss.append(avg_loss)  # Store the average loss
        if (i) % 10 == 0:
            print(f"iterasjon {i}/{n_iter}, Average Loss: {avg_loss}")  # printer per 

    return loss


def accuracy_sorting(neuralnet, x, y, m, r):
    for i in range(r):
        X = onehot(x, m)
        Z = neuralnet.forward(X)

        z_hat = (np.argmax(Z, axis=1)[:, -1]).reshape(-1, 1)
        x = np.concatenate((x, z_hat), axis=1)
    y_hat = x[:, -r:]
    amount = 0
    for i in range(y.shape[0]):
        if np.array_equal(y[i], y_hat[i]):
            amount += 1
    return amount / y.shape[0], y_hat


def accuracy_addition(neuralnet, x, y, m, r):
    for i in range(r+1): #should possibly be r+1
        X = onehot(x, m)
        Z = neuralnet.forward(X)
        z_hat = (np.argmax(Z, axis=1)[:, -1]).reshape(-1, 1)
        x = np.concatenate((x, z_hat), axis=1)
    y_hat = x[:, -(r + 1) :]
    y_hat = [row[::-1] for row in y_hat]
    amount = 0
    for i in range(y.shape[0]):
        if np.array_equal(y[i], y_hat[i]):
            amount += 1
    return amount / y.shape[0], y_hat
