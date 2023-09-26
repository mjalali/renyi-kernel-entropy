from functools import partial

import numpy as np


def gaussian_kernel(x, y, sigma):
    """
    Creates gaussian kernel with sigma bandwidth

    Args:
        x: The first vector
        y: The second vector
        sigma: Gaussian kernel bandwidth

    Returns: Gaussian distance of x and y

    """
    kernel = np.exp(-0.5 * np.sum(np.square(x - y)) / sigma ** 2)
    return kernel


class RKE:
    def __init__(self, kernel_bandwidth=None, kernel_function=None):
        """
        Define the kernel for computing the Renyi Kernel Entropy score

        Args:
            kernel_bandwidth: The bandwidth to use in gaussian_kernel.
                You should pass either of the `kernel_function` or `kernel_bandwidth` arguments.
            kernel_function: The Kernel function to build kernel matrix,
                default is gaussian_kernel as used in the paper.
        """
        if kernel_function is None and kernel_bandwidth is None:
            raise ValueError('Expected either kernel_function or kernel_bandwidth args')
        if kernel_function is None:
            kernel_function = partial(gaussian_kernel, sigma=kernel_bandwidth)
        self.kernel_function = kernel_function

    def predict_n_clusters(self, X, **kwargs):
        f_norm = 0
        for i in range(X.shape[0]):  # TODO: use torch tensors instead of numpy array
            for j in range(X.shape[0]):
                f_norm += self.kernel_function(X[i], X[j])**2
        return f_norm / X.shape[0]**2

    def compute_rke_mc(self, X, n_samples=1024):
        """
        Computing RKE-MC = exp(-RKE(X))
        Args:
            X:
            n_samples: How many samples to compute k(x_i, x_j)

        Returns:
        """
        i = np.random.randint(X.shape[0], size=n_samples)
        j = np.random.randint(X.shape[0], size=n_samples)
        kernel_function_holder = self.kernel_function(X[i], X[j])**2
        print(kernel_function_holder)
        return kernel_function_holder
        # return 1/np.mean(kernel_function_holder)

    def compute_relative_kernel(self, X, Y):
        output = np.ndarray((X.shape[0], Y.shape[0]))

        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                output[i][j] = self.kernel_function(X[i], Y[j])
        return output / np.sqrt(X.shape[0] * Y.shape[0])

    def compute_e_kernel_xy(self, X, Y=None, n_samples=1024):
        if Y is None:
            Y = X
        kernel_function_holder = []
        for i in range(n_samples):
            i, j = np.random.randint(X.shape[0], size=2)

            kernel_function_holder.append(self.kernel_function(X[i], Y[j]))
        return np.mean(kernel_function_holder)

    def compute_kid(self, X, Y, x_samples=500, y_samples=None, kernel_samples=1000):
        if y_samples is None:
            y_samples = x_samples

        self.k_xx = self.compute_e_kernel_xy(X[:x_samples], X[:x_samples], kernel_samples)
        self.k_xy = self.compute_e_kernel_xy(X[:x_samples], Y[:y_samples], kernel_samples)
        self.k_yy = self.compute_e_kernel_xy(Y[:y_samples], Y[:y_samples], kernel_samples)

        return self.k_xx + self.k_yy + - 2 * self.k_xy

    def compute_rrke(self, X, Y, x_samples=500, y_samples=None):
        if y_samples is None:
            y_samples = x_samples

        self.k_xy = self.compute_relative_kernel(X[:x_samples], Y[:y_samples])
        svds = np.linalg.svd(self.k_xy, compute_uv=False)
        sum_svds = np.sum(svds)
        return -np.log(sum_svds**2)
