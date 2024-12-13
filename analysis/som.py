"""Forked from https://github.com/nmarincic/numbasom with adapt() modified to weigh dimensions, added PCA initialization"""

from numbasom.core import SOM, random_lattice, get_all_BMU_indexes, normalize_data, timer
from numba import jit
import math
import numpy as np
import numpy as np


class SOM(SOM):

    def __init__(self, som_size, is_torus, balance=0.5):
        super().__init__(som_size, is_torus)
        self.balance = balance

    def train(self, data, num_iterations, initialize='random', normalize=False, start_lrate = 0.1):
        """Trains the algorithm and returns the lattice.

        If `normalize` is False, there will be no normalization of the input data.

        Parameters
        ---
        data : numpy array

            The input data tensor of the shape NxD, where:
            N - instances axis
            D - features axis

        num_iterations : int

            The number of iterations the algorithm will run.

        normalize : boolean, optional

            If True, the data will be normalized

        Returns
        --
        The lattice of the shape (R,C,D):

        R - number of rows; C - number of columns; D - features axis
        """
        data_scaled = data
        if normalize:
            start = timer()
            data_scaled = normalize_data(data)
            end = timer()
            print("Data scaling took: %f seconds." %(end - start))
        start = timer()
        initial_lattice = initialize_lattice(self.som_size, data_scaled, initialize)
        lattice = som_calc(initial_lattice, self.som_size, num_iterations, data_scaled, start_lrate, self.balance, self.is_torus)
        end = timer()
        print("SOM training took: %f seconds." %(end - start))
        return lattice



def pca(X, num_components):
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    indices = np.argsort(eigenvalues)[::-1][:num_components]
    components = eigenvectors[:, indices]
    return np.dot(X_centered, components), components, eigenvalues[indices]


def reconstruct_from_embedding(embedding, pcs, mean):
    return embedding @ pcs.T + mean


def pca_init(x_mat, som_size):
    transformed_data, pcs, _ = pca(x_mat, 2)
    x_range = np.linspace(np.percentile(transformed_data[:,0], 10), np.percentile(transformed_data[:,0], 90), som_size[0])
    y_range = np.linspace(np.percentile(transformed_data[:,1], 10), np.percentile(transformed_data[:,1], 90), som_size[1])
    embedded_initial_units = np.array([[x, y] for x in x_range for y in y_range])
    raw_initial_units = reconstruct_from_embedding(embedded_initial_units, pcs, x_mat.mean(axis=0))
    return raw_initial_units.reshape(som_size[0], som_size[1], -1)


def initialize_lattice(som_size, data, how='random'):
    if how == 'pca':
        print ("Initializing SOM with PCA")
        x = pca_init(data, som_size)
    elif how =='random':
        print ("Initializing SOM with Random")
        x = random_lattice(som_size, data.shape[1])
    else:
        assert (False)
    print ("Done Init")
    return x


@jit(nopython=True)
def som_calc(lattice, som_size, num_iterations, data, start_lrate, balance, is_torus=False):
    initial_radius = (max(som_size[0],som_size[1])/2)**2
    time_constant = num_iterations/math.log(initial_radius)
    
    datalen = len(data)
    X, Y, Z = lattice.shape

    for current_iteration in range(num_iterations):
        current_radius = initial_radius * math.exp(-current_iteration/time_constant)
        current_lrate = start_lrate * math.exp(-current_iteration/num_iterations)
        rand_input = np.random.randint(datalen)
        rand_vector = data[rand_input]

        BMU_dist = 1.7976931348623157e+308
        BMU = (0,0)

        for x in range(X):
            for y in range(Y):
                d = 0.0
                for z in range(Z):
                    val = lattice[x,y,z]-rand_vector[z]
                    valsqr = val * val
                    d += valsqr

                if d < BMU_dist:
                    BMU_dist = d
                    BMU = (x,y)

        if is_torus:
            BMUs = get_all_BMU_indexes(BMU, X, Y)

            for BMU in BMUs:
                adapt(lattice, rand_vector, BMU, current_radius, current_lrate, balance)

        else:
            adapt(lattice, rand_vector, BMU, current_radius, current_lrate, balance)

    return lattice


@jit(nopython=True)
def adapt(lattice, rand_vector, BMU, current_radius, current_lrate, balance):
    X, Y, Z = lattice.shape
    for x in range(X):
        for y in range(Y):
            a = (x-BMU[0])
            b = (y-BMU[1])
            d = balance*(a**2) + (1-balance)*(b**2)
            if d < current_radius:
                up = d * d
                down = current_radius * current_radius
                res = -up / (2 * down)
                influence = math.exp(res)
                for z in range(Z):
                    diff = (rand_vector[z] - lattice[x,y,z]) * influence * current_lrate
                    lattice[x,y,z] += diff
