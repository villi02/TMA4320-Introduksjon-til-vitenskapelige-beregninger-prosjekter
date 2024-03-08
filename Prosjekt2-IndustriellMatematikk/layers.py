import numpy as np
from utils import onehot
#from neural_network import NeuralNetwork

class Layer:

    """
    Base class for layers in the neural network with forward and backward pass.
    """
    def __init__(self):
        
        return

    def forward(self,inputs):
        raise NotImplementedError

    def backward(self,grad):
        raise NotImplementedError
    
    def step_adam(self, iter, alpha = 0.01, beta1 = 0.9, beta2 = 0.999, epsilon = 10**(-8)):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                'V': V,         The gradient matrix 
                'M': M,         The gradient matrix 
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            G = self.params[param]['d']
            self.params[param]['M'] = beta1*self.params[param]['M'] + (1-beta1)*G
            self.params[param]['V'] = beta2*self.params[param]['V']+(1-beta2)*(G*G)
            M_hat = (1/(1-beta1**iter))*self.params[param]['M']
            V_hat = (1/(1-beta2**iter))*self.params[param]['V']
            self.params[param]['w'] = self.params[param]['w'] - alpha*(M_hat/(np.sqrt(V_hat) +epsilon))



        return
    
    def step_gd(self,alpha):
        """
        Performs a gradient descent step given learning rate.
        Assumes that the layer has a parameter dictionary "params" on the form

        params = {
            'w1': {         
                'w': w,         The parameter matrix
                'd': d,         The gradient of loss wrt the parameter matrix
                },
            'w2': {....},
            
        }
        where each parameter has a key 'w' for weights and 'd' for gradients.
        """
        for param in self.params:
            self.params[param]['w'] -= alpha*self.params[param]['d']
 


class Attention(Layer):

    def __init__(self,d , k):
        """
        Your code here
        """
        self.W_O = LinearLayer(d,k)
        self.W_V = LinearLayer(d,k)
        self.W_K = LinearLayer(d,k)
        self.W_Q = LinearLayer(d,k)
        self.params = {"W_O": {'w': self.W_O.w, 'd': np.zeros_like(self.W_O.w), 'V': np.zeros_like(self.W_O.w), 'M': np.zeros_like(self.W_O.w)},
                       "W_V": {'w': self.W_V.w, 'd': np.zeros_like(self.W_V.w), 'V': np.zeros_like(self.W_V.w), 'M': np.zeros_like(self.W_V.w)},
                       "W_K": {'w': self.W_K.w, 'd': np.zeros_like(self.W_K.w), 'V': np.zeros_like(self.W_K.w), 'M': np.zeros_like(self.W_K.w)},
                       "W_Q": {'w': self.W_Q.w, 'd': np.zeros_like(self.W_Q.w), 'V': np.zeros_like(self.W_Q.w), 'M': np.zeros_like(self.W_Q.w)}}

        self.softmax = Softmax()
        return

        

    def forward(self,x):
        """
        Your code here
        """
        

        n = x.shape[2]
        self.x = x
        self.D = np.zeros((n, n))
        i1,i2 = np.tril_indices(n,-1)
        self.D[i1,i2] = -np.inf #creates D matrix
        self.A = self.softmax.forward(np.einsum('aij,jn,nk,bkt->it', np.transpose(x,(0,2,1)), np.transpose(self.params["W_Q"]['w']), self.params["W_K"]['w'], x) + self.D)
        self.z_nxt = x + np.einsum('in, nj, ajk,kt->aik',np.transpose(self.params["W_O"]['w']), self.params["W_V"]['w'], x, self.A)
        return self.z_nxt   


    def backward(self,grad):
        """
        Your code here
        """
        grad_OV = np.einsum('ab,bc,kcd -> kad',np.transpose(self.params["W_V"]['w']),self.params["W_O"]['w'], grad )
        grad_S = self.softmax.backward(np.einsum('abc, dce ->dbe',np.transpose(self.x,(0,2,1)),grad_OV))
        del_L = grad + np.einsum('abc, ce ->abe', grad_OV, np.transpose(self.A))+np.einsum('ab,bc, kcd, lde -> lae', np.transpose(self.params["W_K"]['w']), self.params["W_Q"]['w'], self.x, grad_S)
        del_L += np.einsum('ab,bc, kcd, lde -> lae', np.transpose(self.params["W_Q"]['w']), self.params["W_K"]['w'], self.x, np.transpose(grad_S,(0,2,1)))

        self.params["W_O"]['d'] = np.einsum('ab, kbc, cd, mde -> ae',self.params["W_V"]['w'], self.x, self.A, np.transpose(grad, (0,2,1)))
        self.params["W_V"]['d'] = np.einsum('ab, kbc, cd, mde -> ae',self.params["W_O"]['w'], grad, np.transpose(self.A), np.transpose(self.x, (0,2,1)))
        self.params["W_K"]['d'] = np.einsum('ab, kbc, lcd, mde -> ae',self.params["W_Q"]['w'], self.x, grad_S, np.transpose(self.x, (0,2,1)))
        self.params["W_Q"]['d'] = np.einsum('ab, kbc, lcd, mde -> ae',self.params["W_K"]['w'], self.x, np.transpose(grad_S, (0,2,1)), np.transpose(self.x, (0,2,1)))

        return del_L
    


class Softmax(Layer):

    def __init__(self):
        """
        Your code here
        """
        return

    
    def forward(self,x):
        """
        Your code here
        """
        self.x = x
        self.Z = np.zeros(x.shape)
        self.epsilon = 10**(-8)
        self.P = np.exp(x)
        Q = np.sum(self.P,axis=1,keepdims=True)
        self.Z = self.P/(Q+self.epsilon)
        return self.Z


    def backward(self,grad):
        """
        Your code here
        """
        self.b = np.zeros(self.x.shape)
        P = np.exp(self.x)
        Q = np.sum(self.P,axis=0,keepdims=True)
        S = P/(Q*Q+self.epsilon)
        self.b = grad*self.Z- np.sum(grad*S)*P     

        return self.b
        



class CrossEntropy(Layer):

    def __init__(self):
        """
        Your code here
        """
        self.epsilon = 10**(-8)
        return

        

    def forward(self, x, y):
        """
        Your code here
        """
        self.x = x
        self.Y_hat = x[:,:,-6:] #husk slice
        self.Y = onehot(y, x.shape[1])
        ones = np.ones(x.shape[1])
        self.P = np.transpose(ones)@(self.Y_hat*self.Y)
        self.Q = -np.log(self.P)
        D = x.shape[0]
        self.n = x.shape[2]
        L=0
        for j in range(D-1):
            for i in range(self.n-1):    
                L += self.Q[i][j]
        L = L/(D*self.n)
        return L


    def backward(self):
        """
        Your code here
        """
        epsilon = 10**(-8)
        del_loss = (-1/self.n)*(self.Y/(self.Y_hat+epsilon))
        pad_width = [(0,0), (0,0), (0,1)]
        del_loss = np.pad(del_loss, pad_width, mode = 'constant') #adds row of 0 to adjust for shape in softmax
        return del_loss
    


class LinearLayer(Layer):

    """
    Linear Layer
    """
    def __init__(self,input_size, output_size,init_scale = 0.1):
        """
        Constructor takes input size and output size of layer 
        and scale for the weights
        """

        #Initialize weights using a sample from the normal distribution
        #scaled with the init_scale
        self.w = np.random.randn(output_size,input_size)*init_scale
        self.params = {"w":{'w':self.w,
                            'd':np.zeros_like(self.w), 'V': np.zeros_like(self.w), 'M': np.zeros_like(self.w)}}
        

    def forward(self,x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x
        
        #Return output of layer
        #y = w@x
        y = np.einsum('od,bdn->bon',self.params['w']['w'],x)
        return y
        
    def backward(self,grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt weight w: 
        #dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params['w']['d'] = np.einsum('bon,bdn->od',grad,self.x)/b

        #Return gradient of loss wrt input of layer
        #dL/dw = w@grad.T
        return np.einsum('od,bon->bdn',self.params['w']['w'],grad)
    

class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self,x):
        #relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self,x):
        
        #Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self,grad):

        #dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))



class EmbedPosition(Layer):
    def __init__(self,n_max,m,d,init_scale=1e-1):   

        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        #Initialize a linear layer for the embedding
        self.embed = LinearLayer(m,d,init_scale)
        #Initialize the position embedding matrix
        self.w = np.random.randn(d,n_max)*init_scale

        #Initialize the parameter dictionary for weight with key "Wp"
        self.params = {"Wp":{'w':self.w,'d':None, 'V': np.zeros_like(self.w), 'M': np.zeros_like(self.w)}}

    def forward(self,X):

        """
        Input:
            X: one-hot encoded array of shape (b,m,n).

        Output:
            z_0: array of shape (b,d,n)

        embed.forward(X) maps (b,m,n) to (b,d,n). 
        Assigns a column of size d to each integer in the sequence
        and add positional embedding matrix (params['Wp']['w'][:,:n]) (b,d,n).

        Equivalent to 

        z_0 = W_E@X + W_P[:,:n]

        """

        #We assume that n < n_max
        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params['Wp']['w'][:,:n]
        return z_0
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        
        b = grad.shape[0]

        #Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params['Wp']['d'] = np.zeros_like(self.w)
        self.params['Wp']['d'] += np.sum(grad,axis=0)/b

        #Use backwards pass of the linear layer
        self.embed.backward(grad)

        #This is always the final layer, so we return None
        return None
    
    def step_gd(self,step_size):

        #We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        #And since we override step_gd(), we use super 
        #which calls the step_gd() of the base class
        #and does gd for the paramters in the params dict
        super().step_gd(step_size)




class FeedForward(Layer):


    def __init__(self,d, p,init_scale = 0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        #first linear layer with input size d and output size p
        self.l1 = LinearLayer(d,p,init_scale)

        #We use the Relu activation function
        self.activation = Relu()

        #second linear layer with input size p and output size d
        self.l2 = LinearLayer(p,d,init_scale)


    def forward(self,x):
        """
        Input:
            - x of shape (b,d,n)
        Output:
            - shape (b,d,n)

        This is equivalent to
        y = x + W2.T@Relu(W1@x)

         (W1,W2 are p x d)
        """

        self.x = x

        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
    
    def backward(self,grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)
        
        """

        #We use backward pass of the linear layers and activation.
        #Recall that the backward pass reverse the order of the layers.
        grad_feed_forward = self.l1.backward(self.activation.backward(self.l2.backward(grad)))

        #Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward


    def step_gd(self,step_size):

        #Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)