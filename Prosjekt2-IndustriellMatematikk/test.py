import unittest

from neural_network import NeuralNetwork
import layers
import tensorflow as tf
from utils import onehot
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.models import Model
from data_generators import get_train_test_sorting
import unittest
import numpy as np


# Assuming Attention, LinearLayer, and Softmax are defined in attention_module
class AttentionTF(Layer):

    def __init__(self, d, k, **kwargs):
        super(layers.Attention, self).__init__(**kwargs)
        # Initializing weights
        self.W_O = self.add_weight(
            name="W_O", shape=(k, d), initializer="random_normal", trainable=True
        )
        self.W_V = self.add_weight(
            name="W_V", shape=(k, d), initializer="random_normal", trainable=True
        )
        self.W_K = self.add_weight(
            name="W_K", shape=(k, d), initializer="random_normal", trainable=True
        )
        self.W_Q = self.add_weight(
            name="W_Q", shape=(k, d), initializer="random_normal", trainable=True
        )
        self.softmax = Softmax()

    def call(self, inputs):
        # Forward pass
        n = tf.shape(inputs)[2]
        b = tf.shape(inputs)[0]
        x_transpose = tf.transpose(inputs, perm=[0, 2, 1])

        D = tf.zeros((n, n))
        i1, i2 = tf.linalg.band_part(tf.ones((n, n)), -1, 0) - 1
        D = tf.where(i1 == 0, x=tf.constant(-float("inf")), y=D)

        A = self.softmax(
            tf.einsum("bij,jn,nk,bkt->bit", x_transpose, self.W_Q, self.W_K, inputs) + D
        )
        output = inputs + tf.einsum(
            "in, nj, ajk,akt->aik", self.W_O, self.W_V, inputs, A
        )

        return output


class TestAttention(unittest.TestCase):

    def setUp(self):
        # Dimensions for the test
        self.d = 4  # Dimension of input
        self.k = 3  # Dimension of output
        self.batch_size = 2
        self.seq_length = 5

        # Initialize the Attention layer
        self.attention = layers.Attention(self.d, self.k)

        # Generate a random input of shape (batch_size, d, seq_length)
        self.input = np.random.rand(self.batch_size, self.d, self.seq_length)

        # Generate a random gradient of the same shape as input for backward pass
        self.grad_output = np.random.rand(self.batch_size, self.d, self.seq_length)

    def test_forward_backward(self):
        # Forward pass
        output = self.attention.forward(self.input)
        self.assertEqual(output.shape, (self.batch_size, self.d, self.seq_length))

        # Backward pass
        grad_input = self.attention.backward(self.grad_output)
        self.assertEqual(grad_input.shape, (self.batch_size, self.d, self.seq_length))

    def test_parameter_update(self):
        # Perform a forward and backward pass
        self.attention.forward(self.input)
        self.attention.backward(self.grad_output)

        # Store old parameters for comparison
        old_params = {
            key: param["w"].copy() for key, param in self.attention.params.items()
        }

        # Simple gradient descent update
        learning_rate = 0.01
        for param in self.attention.params.values():
            param["w"] -= learning_rate * param["d"]

        # Check if parameters are updated (not equal to old parameters)
        for key, old_w in old_params.items():
            new_w = self.attention.params[key]["w"]
            with self.subTest(param=key):
                self.assertFalse(
                    np.array_equal(old_w, new_w), f"Parameter {key} was not updated"
                )


class TestAttentionLayer:
    def setUp(self):
        r = 5
        m = 2
        d = 10
        k = 5
        p = 15
        L = 2
        n_max = 2 * r - 1
        n_iter = 300
        alpha = 0.001
        num_of_samples = 250
        num_train_batches = 10
        num_test_batches = 1
        data = get_train_test_sorting(
            r, m, num_of_samples, num_train_batches, num_test_batches
        )
        self.x = data["x_train"]
        self.y = data["y_train"]
        self.custom_attention_layer = layers.Attention(d=10, k=5)
        self.tf_attention_layer = AttentionTF(d=10, k=5)

    def test_attention_output(self):
        # Process input through custom attention layer
        custom_output = self.custom_attention_layer.forward(self.x)

        # Process input through TensorFlow attention layer
        # Note: This assumes AttentionTF is integrated within a TensorFlow model
        inputs = Input(shape=(None, self.x.shape[-1]))
        attention_output = AttentionTF(d=10, k=5)(inputs)
        model = Model(inputs=inputs, outputs=attention_output)
        tf_output = model.predict(self.x)

        # Verify outputs are close enough
        np.testing.assert_almost_equal(custom_output, tf_output, decimal=5)


class TestSoftmaxLayer(unittest.TestCase):
    def setUp(self):
        self.softmax = layers.Softmax()
        self.input_data = np.random.randn(3, 5)  # Batch size of 3, 5 classes
        self.grad_output = np.random.randn(3, 5)

    def test_forward_output(self):
        output = self.softmax.forward(self.input_data)
        # Check if softmax output is correctly normalized
        for row in output:
            self.assertAlmostEqual(np.sum(row), 1.0)

    def test_forward_output_example(self):
        softb = layers.Softmax()
        example_input = np.array(
            [[-4.1, 2.2, 3.0, -0.1], [0.1, 0.1, 1.2, 0.3], [-3.6, -1.1, 3.9, -0.1]]
        )
        output = softb.forward(example_input)
        expected_output = tf.nn.softmax(example_input)
        self.assertTrue(np.allclose(output, expected_output, atol=1e-5))

    def test_backward_shape(self):
        self.softmax.forward(self.input_data)  # Forward pass to set up for backward
        grad_input = self.softmax.backward(self.grad_output)
        self.assertEqual(grad_input.shape, self.input_data.shape)

    def test_gradients(self):
        self.softmax.forward(self.input_data)
        grad_input = self.softmax.backward(self.grad_output)
        # This is a basic check. In practice, you might want to check the correctness of the gradient values more thoroughly.
        self.assertFalse(np.array_equal(grad_input, np.zeros_like(grad_input)))


class TestCrossEntropyLayer(unittest.TestCase):
    def setUp(self):
        self.cross_entropy = layers.CrossEntropy()
        r = 5
        self.m = 2
        d = 10
        k = 5
        p = 15
        L = 2
        n_max = 2 * r - 1
        n_iter = 300
        alpha = 0.001
        num_of_samples = 250
        num_train_batches = 10
        num_test_batches = 1
        data = get_train_test_sorting(
            r, self.m, num_of_samples, num_train_batches, num_test_batches
        )
        self.x = data["x_train"]
        self.y = data["y_train"]

    def test_forward_loss_against_tf(self):
        X_batch = onehot(self.x[0], self.m)
        Z = layers.Softmax().forward(X_batch)
        loss = self.cross_entropy.forward(Z, self.y[0])
        expected_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(self.y[0], Z)
        )
        self.assertAlmostEqual(loss, expected_loss)

    def test_backward_shape(self):
        X_batch = onehot(self.x[0], self.m)

        self.cross_entropy.forward(self.x, self.y[0])
        grad = self.cross_entropy.backward()
        # Ensure gradient shape matches the input x shape
        self.assertEqual(grad.shape, self.x.shape)


if __name__ == "__main__":
    unittest.main()
