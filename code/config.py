# system of three first-order differential equations
def dx(x, y, z):
    return y + z


def dy(x, y, z):
    return -x + 0.5 * y


def dz(x, y, z):
    return x ** 2 - z


# hyperparameters for neural networks objects:
input_layer_size_0 = 3
input_layer_size_1 = 6

hidden_layer_size_0 = 12
hidden_layer_size_1 = 12

alpha_0 = 0.05
alpha_1 = 0.05

Em_0 = 0.05
Em_1 = 0.05

# dataset configuration:
ds_size = 1500

learn_size = 0.5
valid_size = 0.5
test_size = 0

h = 0.1
