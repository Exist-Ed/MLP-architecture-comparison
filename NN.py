import numpy as np
from time import perf_counter


class MLP:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size=3, alpha=0.5, Em=0.001):
        self.alpha = alpha
        self.Em = Em
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

        self.W_hidden = np.random.random(input_layer_size * hidden_layer_size).reshape(
            (hidden_layer_size, input_layer_size))
        self.T_hidden = np.random.random(hidden_layer_size).reshape((hidden_layer_size, 1))

        self.W_output = np.random.random(output_layer_size * hidden_layer_size).reshape(
            (output_layer_size, hidden_layer_size))
        self.T_output = np.random.random(output_layer_size).reshape((output_layer_size, 1))

    def lin_foo(self, S):
        return S

    def d_lin_foo(self, S):
        return 1

    def sigm_foo(self, S):
        return 1 / (1 + np.exp(-S))

    def d_sigm_foo(self, S):
        return (1 - S) * S

    def predict(self, X):
        X = np.ravel(X).reshape((self.input_layer_size, 1))

        Y_hidden = self.W_hidden @ X
        Y_hidden -= self.T_hidden
        Y_hidden = np.apply_along_axis(self.sigm_foo, 0, Y_hidden)

        Y_output = self.W_output @ Y_hidden
        Y_output -= self.T_output
        Y_output = np.apply_along_axis(self.lin_foo, 0, Y_output)

        return np.ravel(Y_output)

    def fit(self, X):
        t_start = perf_counter()
        current_error = float('inf')

        while current_error > self.Em:
            err = []
            for i in range(int(self.input_layer_size / 3), len(X)):
                input_x = np.ravel(X[i - int(self.input_layer_size / 3):i]).reshape((self.input_layer_size, 1))

                Y_hidden = self.W_hidden @ input_x
                Y_hidden -= self.T_hidden
                Y_hidden = np.apply_along_axis(self.sigm_foo, 0, Y_hidden)

                Y_output = self.W_output @ Y_hidden
                Y_output -= self.T_output
                Y_output = np.apply_along_axis(self.lin_foo, 0, Y_output)

                err.append(np.sum((Y_output.T - X[i]) ** 2))

                e_output = Y_output - X[i].reshape((self.output_layer_size, 1))
                e_hidden = self.W_output.T @ e_output

                self.W_output -= self.alpha * (e_output @ Y_hidden.T)
                self.T_output += self.alpha * e_output

                self.W_hidden -= (self.alpha * e_hidden * np.apply_along_axis(self.d_sigm_foo, 0,
                                                                              Y_hidden)) @ input_x.T
                self.T_hidden += self.alpha * e_hidden * np.apply_along_axis(self.d_sigm_foo, 0, Y_hidden)

            current_error = float(0.5 * sum(err))
            print(current_error)

        with open('timer_logs.txt', 'a+') as file:
            file.write(str(divmod(perf_counter() - t_start, 60)) + '\n')
