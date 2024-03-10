import unittest

from neural_network import NeuralNetwork
from layers import *

import unittest
import numpy as np
# Assuming Attention, LinearLayer, and Softmax are defined in attention_module

class TestAttention(unittest.TestCase):

    def setUp(self):
        # Dimensions for the test
        self.d = 4  # Dimension of input
        self.k = 3  # Dimension of output
        self.batch_size = 2
        self.seq_length = 5
        
        # Initialize the Attention layer
        self.attention = Attention(self.d, self.k)

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
        old_params = {key: param["w"].copy() for key, param in self.attention.params.items()}
        
        # Simple gradient descent update
        learning_rate = 0.01
        for param in self.attention.params.values():
            param["w"] -= learning_rate * param["d"]
        
        # Check if parameters are updated (not equal to old parameters)
        for key, old_w in old_params.items():
            new_w = self.attention.params[key]["w"]
            with self.subTest(param=key):
                self.assertFalse(np.array_equal(old_w, new_w), f"Parameter {key} was not updated")

class TestSoftmaxLayer(unittest.TestCase):
    def setUp(self):
        self.softmax = Softmax()
        self.input_data = np.random.randn(3, 5)  # Batch size of 3, 5 classes
        self.grad_output = np.random.randn(3, 5)

    def test_forward_output(self):
        output = self.softmax.forward(self.input_data)
        # Check if softmax output is correctly normalized
        for row in output:
            self.assertAlmostEqual(np.sum(row), 1.0)

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
        self.cross_entropy = CrossEntropy()
        self.num_classes = 6  # Assume 6 classes for simplicity
        self.batch_size = 3
        self.seq_length = 10  # Assuming a sequence length or feature dimension of 10
        self.x = np.random.randn(self.batch_size, self.seq_length, self.num_classes)
        self.y = np.random.randint(0, self.num_classes, size=(self.batch_size,))

    def test_forward_loss(self):
        loss = self.cross_entropy.forward(self.x, self.y)
        # Basic check to ensure loss is calculated and is a scalar
        self.assertIsInstance(loss, float)

    def test_backward_shape(self):
        self.cross_entropy.forward(self.x, self.y)
        grad = self.cross_entropy.backward()
        # Ensure gradient shape matches the input x shape
        self.assertEqual(grad.shape, self.x.shape)

    def test_backward_values(self):
        self.cross_entropy.forward(self.x, self.y)
        grad = self.cross_entropy.backward()
        # Check that gradients exist and are not all zeros
        self.assertFalse(np.all(grad == 0))



if __name__ == "__main__":
    unittest.main()