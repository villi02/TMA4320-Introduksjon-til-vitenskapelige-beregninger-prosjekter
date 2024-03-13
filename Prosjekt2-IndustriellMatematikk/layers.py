import numpy as np
from utils import onehot

# from neural_network import NeuralNetwork


class Layer:
    """
    Base class for layers in the neural network with forward and backward pass.
    """

    def __init__(self):

        return

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def step_adam(self, iter, alpha=0.01, beta1=0.9, beta2=0.999, epsilon=10 ** (-8)):
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
            G = self.params[param]["d"]
            self.params[param]["M"] = beta1 * self.params[param]["M"] + (1 - beta1) * G
            self.params[param]["V"] = beta2 * self.params[param]["V"] + (1 - beta2) * (
                G * G
            )
            M_hat = (1 / (1 - beta1**iter)) * self.params[param]["M"]
            V_hat = (1 / (1 - beta2**iter)) * self.params[param]["V"]
            self.params[param]["w"] = self.params[param]["w"] - alpha * (
                M_hat / (np.sqrt(V_hat) + epsilon)
            )

        return

    def step_gd(self, alpha):
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
            self.params[param]["w"] -= alpha * self.params[param]["d"]


class Attention(Layer):

    def __init__(self, d, k):
        """
        Your code here
        """

        self.W_O = np.random.randn(k, d) * 0.1
        self.W_V = np.random.randn(k, d) * 0.1
        W_K = np.random.randn(k, d) * 0.1
        W_Q = np.random.randn(k, d) * 0.1
        self.params = {
            "W_O": {
                "w": self.W_O,
                "d": np.zeros_like(self.W_O),
                "V": np.zeros_like(self.W_O),
                "M": np.zeros_like(self.W_O),
            },
            "W_V": {
                "w": self.W_V,
                "d": np.zeros_like(self.W_V),
                "V": np.zeros_like(self.W_V),
                "M": np.zeros_like(self.W_V),
            },
            "W_K": {
                "w": W_K,
                "d": np.zeros_like(W_K),
                "V": np.zeros_like(W_K),
                "M": np.zeros_like(W_K),
            },
            "W_Q": {
                "w": W_Q,
                "d": np.zeros_like(W_Q),
                "V": np.zeros_like(W_Q),
                "M": np.zeros_like(W_Q),
            },
        }
        self.softmax = Softmax()
        return

    def forward(self, x):
        """
        Your code here
        """

        n = x.shape[2]
        self.b = x.shape[0]
        self.x = x
        self.x_transpose = np.transpose(self.x, (0, 2, 1))

        self.D = np.zeros((n, n))
        i1, i2 = np.tril_indices(n, -1)
        self.D[i1, i2] = -np.inf  # creates D matrix

        self.A = self.softmax.forward(
            np.einsum(
                "bij, jn, nk, bkt->bit",
                self.x_transpose,
                np.transpose(self.params["W_Q"]["w"]),
                self.params["W_K"]["w"],
                x,
                optimize=True,
            )
            + self.D
        )

        return self.x + np.einsum(
            "in, nj, ajk, akt -> ait",
            np.transpose(self.params["W_O"]["w"]),
            self.params["W_V"]["w"],
            self.x,
            self.A,
            optimize=True,
        )

    def backward(self, grad):
        """
        Your code here
        """

        # Useful calculations

        grad_OV = np.einsum(
            "ab, bc, kcd -> kad",
            np.transpose(self.params["W_V"]["w"]),
            self.params["W_O"]["w"],
            grad,
            optimize=True,
        )
        grad_S = self.softmax.backward(
            np.einsum(
                "abc, ace ->abe",
                self.x_transpose,
                grad_OV,
                optimize=True,
            )
        )

        grad_S_T = np.transpose(grad_S, (0, 2, 1))

        self.A_transpose = np.transpose(self.A, (0, 2, 1))

        del_L = grad + np.einsum(
            "abc, ace -> abe", grad_OV, self.A_transpose, optimize=True
        )
        del_L += np.einsum(
            "ab,bc, kcd, lde -> lae",
            np.transpose(self.params["W_K"]["w"]),
            self.params["W_Q"]["w"],
            self.x,
            grad_S,
            optimize=True,
        )
        del_L += np.einsum(
            "ab, bc, lcd, lde -> lae",
            np.transpose(self.params["W_Q"]["w"]),
            self.params["W_K"]["w"],
            self.x,
            grad_S_T,
            optimize=True,
        )

        self.params["W_O"]["d"] = (
            np.einsum(
                "ab, kbc, kcd, kde -> ae",
                self.params["W_V"]["w"],
                self.x,
                self.A,
                np.transpose(grad, (0, 2, 1)),
                optimize=True,
            )
            / self.b
        )
        self.params["W_V"]["d"] = (
            np.einsum(
                "ab, kbc, kcd, kde -> ae",
                self.params["W_O"]["w"],
                grad,
                self.A_transpose,
                self.x_transpose,
                optimize=True,
            )
            / self.b
        )
        self.params["W_K"]["d"] = (
            np.einsum(
                "ab, kbc, kcd, kde -> ae",
                self.params["W_Q"]["w"],
                self.x,
                grad_S,
                self.x_transpose,
                optimize=True,
            )
            / self.b
        )
        self.params["W_Q"]["d"] = (
            np.einsum(
                "ab, kbc, kcd, kde -> ae",
                self.params["W_K"]["w"],
                self.x,
                grad_S_T,
                self.x_transpose,
                optimize=True,
            )
            / self.b
        )
        return del_L


class Softmax(Layer):

    def __init__(self):
        """
        Your code here
        """
        self.epsilon = 10 ** (-8)
        return

    def forward(self, x):
        """
        Your code here
        """
        self.x = x
        self.P = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.Q = np.sum(self.P, axis=1, keepdims=True)
        self.Z = self.P / (self.Q + self.epsilon)
        return self.Z

    def backward(self, grad):
        """
        Your code here
        """

        S = self.P / (self.Q * self.Q + self.epsilon)
        return (
            grad * self.Z
            - np.sum((grad + self.epsilon) * S, axis=1, keepdims=True) * self.P
        )


class CrossEntropy(Layer):

    def __init__(self):
        """
        Your code here
        """
        self.epsilon = 10 ** (-8)
        return

    def forward(self, x, y):
        """
        Your code here
        """
        self.x = x
        self.b, m, self.n = self.x.shape
        self.Y_hat = x[:, :, -y.shape[-1] :]  # husk slice
        self.Y = onehot(y, m)
        P = np.sum((self.Y_hat * self.Y), axis=1)
        Q = -np.log10(P + self.epsilon)
        return np.mean(Q)

    def backward(self):
        """
        Your code here
        """
        Z = np.zeros_like(self.x)
        Z[:, :, -self.Y.shape[-1] :] = self.Y
        del_loss = (-1 / self.n * self.b) * (Z / (self.x + self.epsilon))
        return del_loss


class LinearLayer(Layer):
    """
    Linear Layer
    """

    def __init__(self, input_size, output_size, init_scale=0.1):
        """
        Constructor takes input size and output size of layer
        and scale for the weights
        """

        # Initialize weights using a sample from the normal distribution
        # scaled with the init_scale
        self.w = np.random.randn(output_size, input_size) * init_scale
        self.params = {
            "w": {
                "w": self.w,
                "d": np.zeros_like(self.w),
                "V": np.zeros_like(self.w),
                "M": np.zeros_like(self.w),
            }
        }

    def forward(self, x):
        """
        Computes the affine transformation of the forward pass
        Stores input for backwards pass and returns output y = Wx.

        x: input, array of shape (batch_size, input_size, n) = (b,d,n)
        y: output, array of shape (batch_size, output_size, n) = (b,o,n)
        """

        self.x = x

        # Return output of layer
        # y = w@x
        y = np.einsum("od,bdn->bon", self.params["w"]["w"], x, optimize=True)
        return y

    def backward(self, grad):
        """
        Performs backward pass.

        grad: gradient of loss wrt output of layer, shape (batch_size, output_size, n) = (b,o,n)
        """

        b = grad.shape[0]

        # Compute gradient (average over B batches) of loss wrt weight w:
        # dL/dw = (1/B)*sum_b^B (grad_b@x_b^T)
        self.params["w"]["d"] = (
            np.einsum("bon,bdn->od", grad, self.x, optimize=True) / b
        )

        # Return gradient of loss wrt input of layer
        # dL/dw = w@grad.T

        return np.einsum("od,bon->bdn", self.params["w"]["w"], grad, optimize=True)


class Relu(Layer):
    """
    Relu activation function
    """

    def __init__(self):
        return

    def relu(self, x):
        # relu(x) = max(0,x)
        return np.maximum(np.zeros(x.shape), x)

    def forward(self, x):

        # Store input for backwards pass
        self.x = x
        return self.relu(x)

    def backward(self, grad):

        # dL/dx = grad * relu'(x)
        return grad * np.where(self.x > 0, np.ones_like(self.x), np.zeros_like(self.x))


class EmbedPosition(Layer):
    def __init__(self, n_max, m, d, init_scale=1e-1):
        """
        n_max: maximum length of input sequence
        m: number of items in the vocabulary / number of integers
        d: embedding dimension
        """

        # Initialize a linear layer for the embedding
        self.embed = LinearLayer(m, d, init_scale)
        # Initialize the position embedding matrix
        self.w = np.random.randn(d, n_max) * init_scale

        # Initialize the parameter dictionary for weight with key "Wp"
        self.params = {
            "Wp": {
                "w": self.w,
                "d": None,
                "V": np.zeros_like(self.w),
                "M": np.zeros_like(self.w),
            }
        }

    def forward(self, X):
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

        # We assume that n < n_max

        n = X.shape[-1]
        z_0 = self.embed.forward(X) + self.params["Wp"]["w"][:, :n]
        return z_0

    def backward(self, grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - None
        """

        b = grad.shape[0]

        # Compute gradient (average over B batches) of loss wrt positional embedding w:
        self.params["Wp"]["d"] = np.zeros_like(self.w)
        self.params["Wp"]["d"] += np.sum(grad, axis=0) / b

        # Use backwards pass of the linear layer
        self.embed.backward(grad)

        # This is always the final layer, so we return None

        return None

    def step_gd(self, step_size):

        # We need to call the step_gd method of the linear layer
        self.embed.step_gd(step_size)

        # And since we override step_gd(), we use super
        # which calls the step_gd() of the base class
        # and does gd for the paramters in the params dict
        super().step_gd(step_size)

    def step_adam(self, iter, alpha=0.01):

        # We need to call the step_gd method of the linear layer
        self.embed.step_adam(iter, alpha)

        # And since we override step_gd(), we use super
        # which calls the step_gd() of the base class
        # and does gd for the paramters in the params dict
        super().step_adam(iter, alpha)


class FeedForward(Layer):

    def __init__(self, d, p, init_scale=0.1):
        """
        Input:
            d: input dimension of first layer and output of second
            p: output dimension of first and input of second.

        """

        # first linear layer with input size d and output size p
        self.l1 = LinearLayer(d, p, init_scale)

        # We use the Relu activation function
        self.activation = Relu()

        # second linear layer with input size p and output size d
        self.l2 = LinearLayer(p, d, init_scale)

    def forward(self, x):
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
        z = x + self.l2.forward(self.activation.forward(self.l1.forward(x)))
        return x + self.l2.forward(self.activation.forward(self.l1.forward(x)))

    def backward(self, grad):
        """
        Input:
            - grad of shape (b,d,n)

        Output:
            - derivative of loss wrt input x. Shape (b,d,n)

        """

        # We use backward pass of the linear layers and activation.
        # Recall that the backward pass reverse the order of the layers.
        grad_feed_forward = self.l1.backward(
            self.activation.backward(self.l2.backward(grad))
        )
        # Since forward pass is x + W2.T@Relu(W1@x)
        return grad + grad_feed_forward

    def step_gd(self, step_size):

        # Call the step_gd method of the linear layers
        self.l1.step_gd(step_size)
        self.l2.step_gd(step_size)

    def step_adam(self, iter, alpha=0.01):

        # We need to call the step_gd method of the linear layer
        self.l1.step_adam(iter, alpha)
        self.l2.step_adam(iter, alpha)

        # And since we override step_gd(), we use super
        # which calls the step_gd() of the base class
        # and does gd for the paramters in the params dict
