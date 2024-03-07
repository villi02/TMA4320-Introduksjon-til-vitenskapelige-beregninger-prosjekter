import numpy as np
from utils import onehot

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
        self.W_O = LinearLayer(k, d)
        self.W_V = LinearLayer(k, d)
        self.W_K = LinearLayer(k, d)
        self.W_Q = LinearLayer(k, d)

        self.softmax = Softmax()
        return

        

    def forward(self,x):
        """
        Your code here
        """
        n = x.shape[2]
        self.D = np.zeros(n, n)
        i1,i2 = np.tril_indices(n,-1)
        self.D[i1,i2] -= np.inf #creates D matrix

        A = self.softmax.forward(np.einsum('bni,nij,njk,bnj->bn', x, self.W_Q, self.W_K, x) + self.D)
        self.z_nxt = x + np.transpose(self.W_O)*self.W_V*x*A

        return self.z_nxt


    def backward(self,grad):
        """
        Your code here
        """
        return
    


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
        for i in range(x.shape[1]):
            P = np.exp(x[i]-self.x.max(axis=0,keepdims=True))
            Q = np.sum(self.P,axis=0,keepdims=True)
            self.Z[i] = P/(Q+self.epsilon)
        return self.Z


    def backward(self,grad):
        """
        Your code here
        """
        self.b = np.zeros(self.x.shape)
        for i in range(self.x.shape[1]):
            P = np.exp(self.x[i]-self.x.max(axis=0,keepdims=True))
            Q = np.sum(self.P,axis=0,keepdims=True)
            S = P/(Q*Q+self.epsilon)
            self.b[i] = grad*self.Z[i]- np.sum(grad*S)*P     

        return self.b
        



class CrossEntropy(Layer):

    def __init__(self, D):
        """
        Your code here
        """
        self.D = D #datapunkter av form {x,y}
        self.n = len(D)
        self.epsilon = 10**(-8)
        return

        

    def forward(self, x, y):
        """
        Your code here
        """
        
        self.Y_hat = x[2*len(y)-1:3*len(y)]
        n = self.Y_hat.shape[1]
        D = self.Y_hat.shape[0]
        self.p= np.zeros(n)
        self.L = 0
        self.y = y

        for j in range(D-1):
            for i in range(n-1):
                self.p[i] = self.Y_hat[i][y[i]] #gets away without using Y matrix because we can just use index of y
            self.q = -np.log(self.p)
            self.L += np.sum(self.q)/n

        self.L =self.L/(D)
        return self.L


    def backward(self):
        """
        Your code here
        """
        self.Y = np.zeros(self.Y_hat.shape)
        for i in range(len(self.y)):
            self.Y[i][self.y[i]] = 1 #basicly creates onehot(y)

        self.del_loss = (-1/self.n)*(self.Y/(self.Y_hat+self.epsilon))
        return self.del_loss
    


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
                            'd':np.zeros_like(self.w)}}
        

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
        self.params = {"Wp":{'w':self.w,'d':None}}

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