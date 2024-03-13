import numpy as np
from layers import *
from neural_network import NeuralNetwork
from utils import onehot
def algorithm_4_sort(x, y, n_iter, alpha, m, neuralnet, r):
    batch_size, n_samples, _ = x.shape

    loss = []  # for å plotte
    for i in range(1, n_iter+1): #antall iterasjoner
        total_loss = []
        for k in range(batch_size):
            X_batch = onehot(x[k], m)
            Z = neuralnet.forward(X_batch)
            batch_loss = neuralnet.loss.forward(Z, y[k][:, -r:])
            total_loss.append(batch_loss)
            dLdz = neuralnet.loss.backward()
            neuralnet.backward(dLdz)
            neuralnet.step_adam( i, alpha)  # 

        avg_loss = np.mean(total_loss)  # regner ut gjennomsnittlig loss
        loss.append(avg_loss)  # lagrer gjennomsnittlig loss

        if (i) % 10 == 0:
            print(f"iterasjon {i}/{n_iter}, gjennomsnittlig Loss: {avg_loss}")  # printer per 10. iterasjoner

    return loss

#forskjell på algoritme4 for addisjon og sortering er hvordan den slicer i loss.forward 

def algorithm_4_add(x, y, n_iter, alpha, m, r, neuralnet):
    batch_size, n_samples, _ = x.shape

    loss = []  # liste for å plotte senere
    for i in range(1, n_iter+1): #antall iterasjoner
        total_loss = []
        for k in range(batch_size): #antall batches
            X_batch = onehot(x[k], m)
            Z = neuralnet.forward(X_batch) #regner ut prediksjon

            batch_loss = neuralnet.loss.forward(Z, y[k][:, -(r + 1) :]) #slicer lengden av y fra prediksjonen til modellen
            total_loss.append(batch_loss)

            dLdz = neuralnet.loss.backward()
            neuralnet.backward(dLdz)
            neuralnet.step_adam(i, alpha  )  

        avg_loss = np.mean(total_loss)  # regner ut gjennomsnittlig loss
        loss.append(avg_loss)  #lagrer gjennomsnittlig loss
        if (i) % 10 == 0:
            print(f"iterasjon {i}/{n_iter}, gjennomsnittlig Loss: {avg_loss}")  # printer per 10. iterasjoner

    return loss


def accuracy_sorting(neuralnet, x, y, m, r):
    for i in range(r): #r blir lengde på y, som brukes for sammenligning
        X = onehot(x, m)
        Z = neuralnet.forward(X)

        z_hat = (np.argmax(Z, axis=1)[:, -1]).reshape(-1, 1) #her brukes argmax(Z , axis = 1)[:, -1] for finne hvilken indeks modellen ville valgt,
                                                             #deretter slicer for å bare ta siste z_hat per sample for å legge den til x, før den kjører på nytt
                                                             # .reshape(-1, 1) brukes for å gjøre (250, ) arrayet til et (250, 1) sånn at np.concatenate fungerer
        x = np.concatenate((x, z_hat), axis=1)
    y_hat = x[:, -r:] # tar siste verdier i x som blir lik y_hat
    amount = 0
    for i in range(y.shape[0]):
        if np.array_equal(y[i], y_hat[i]): # sjekker om prediskjon og svar er lik
            amount += 1
    return amount / y.shape[0]


def accuracy_addition(neuralnet, x, y, m, r):
    for i in range(r+1):
        X = onehot(x, m)
        Z = neuralnet.forward(X)
        z_hat = (np.argmax(Z, axis=1)[:, -1]).reshape(-1, 1)
        x = np.concatenate((x, z_hat), axis=1)
    y_hat = x[:, -(r + 1):]  #en av forskjellen fra sortering er slicen her
    y_hat = [row[::-1] for row in y_hat]  #en annen forskjell er at y_hat må reverseres
    amount = 0
    for i in range(y.shape[0]):
        if np.array_equal(y[i], y_hat[i]):
            amount += 1
    return amount / y.shape[0]
