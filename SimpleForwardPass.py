import numpy as np

# The sigmoid function is used to convert outputs to within a scale of 0 to 1. An output with a value close to 1 is synonymous with having a high probability that the estimated output is a good prediction of the actual output, which we would already know from human-derived labels.

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1/(1+np.exp(-x))

# Network size
N_input = 5
N_hidden = 4
N_output = 2

np.random.seed(42)
# Make some fake data
X = np.random.randn(5)

# To produce the first forward pass, we need to establish random weights for the activation function:
weights_input_to_hidden = np.random.normal(0, scale=0.1, size=(N_input, N_hidden))
weights_hidden_to_output = np.random.normal(0, scale=0.1, size=(N_hidden, N_output))


# Make a forward pass through the network. The dot product function multiplies each value of the input vector X with the corresponding weights in the weight matrix. We then convert the activation function using sigmoid :

hidden_layer_in = np.dot(X, weights_input_to_hidden)
hidden_layer_out = sigmoid(hidden_layer_in)

print('Hidden-layer Output:')
print(hidden_layer_out)

output_layer_in = np.dot(hidden_layer_out, weights_hidden_to_output)
output_layer_out = sigmoid(output_layer_in)

print('Output-layer Output:')
print(output_layer_out)
