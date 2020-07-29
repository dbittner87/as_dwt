import numpy as np
import numpy.linalg as la
import scipy.linalg as sla


class BayesianInverseProblem:
    """Class representing a Bayesian inverse problem"""

    def __init__(self, data, invNoiseCovMat, absPdeProblem=None):
        """
        Initializes a Bayesian inverse problem.

        :param data: Data
        :param absPdeProblem: Abstract PDE problem (must implement an 'instantiate' method to get a concrete PDE problem for a specified parameter)
        """
        self.data = data
        self.invNoiseCovMat = invNoiseCovMat
        self.absPdeProblem = absPdeProblem

        self._sqrtInvNoiseCovMat = sla.sqrtm(invNoiseCovMat)

        self.dimData = len(data)

    def misfit(self, x):
        """Computes the data misfit for a specified parameter."""
        pb = self.absPdeProblem.instantiate(x)
        return self.misfitG(pb.getQoI())

    def misfitG(self, G):
        return 0.5 * la.norm(np.dot(self._sqrtInvNoiseCovMat, self.data - G))**2

    def misfitGradient(self, x):
        pb = self.absPdeProblem.instantiate(x)
        return self.misfitGradientG(pb.getQoI(), pb.getJacobian())

    def misfitGradientG(self, G, jacG):
        return np.dot(np.dot(jacG.T, self.invNoiseCovMat), self.data - G)
