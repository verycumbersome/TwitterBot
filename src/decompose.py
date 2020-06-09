import numpy as np
import copy


class PCA():
    """
    PCA. A class to reduce dimensions
    """

    def __init__(self, retain_ratio):
        """

        :param retain_ratio: percentage of the variance we maitain (see slide for definition)
        """
        self.retain_ratio = retain_ratio

    @staticmethod
    def mean(x):
        """
        returns mean of x
        :param x: matrix of shape (n, m)
        :return: mean of x of with shape (m,)
        """
        return x.mean(axis=0)

    @staticmethod
    def cov(x):
        """
        returns the covariance of x,
        :param x: input data of dim (n, m)
        :return: the covariance matrix of (m, m)
        """
        return np.cov(x.T)

    @staticmethod
    def eig(c):
        """
        returns the eigval and eigvec
        :param c: input matrix of dim (m, m)
        :return:
            eigval: a numpy vector of (m,)
            eigvec: a matrix of (m, m), column ``eigvec[:,i]`` is the eigenvector corresponding to the
        eigenvalue ``eigval[i]``
            Note: eigval is not necessarily ordered
        """

        eigval, eigvec = np.linalg.eig(c)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)
        return eigval, eigvec


    def fit(self, x):
        """
        fits the data x into the PCA. It results in self.eig_vecs and self.eig_values which will
        be used in the transform method
        :param x: input data of shape (n, m); n instances and m features
        :return:
            sets proper values for self.eig_vecs and eig_values
        """

        self.eig_vals = None
        self.eig_vecs = None

        x = x - PCA.mean(x)
        n = x.shape[0]

        ########################################
        #       YOUR CODE GOES HERE            #
        ########################################
        covar_matrix = PCA.cov(x)
        eigval, eigvec = PCA.eig(covar_matrix)
        
        eigvec = eigvec[eigval.argsort()[::-1]]
        eigval = eigval[eigval.argsort()[::-1]]

        k=0
        self.eig_vals = np.delete(eigval, np.s_[k:561])
        self.eig_vecs = np.delete(eigvec, np.s_[k:561], axis=1)
        while(np.sum(self.eig_vals) < 0.9*np.sum(eigval)):
            k = k+1
            self.eig_vals = np.delete(eigval, np.s_[k:561])
            self.eig_vecs = np.delete(eigvec, np.s_[k:561], axis=1)
        
        self.eig_vals = np.delete(eigval, np.s_[k-1:561])
        self.eig_vecs = np.delete(eigvec, np.s_[k-1:561], axis=1)

    def transform(self, x):
        """
        projects x into lower dimension based on current eig_vals and eig_vecs
        :param x: input data of shape (n, m)
        :return: projected data with shape (n, len of eig_vals)
        """

        if isinstance(x, np.ndarray):
            x = np.asarray(x)
        if self.eig_vecs is not None:
            return np.matmul(x, self.eig_vecs)
        else:
            return x
