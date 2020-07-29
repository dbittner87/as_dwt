from . import mcmc
from .stats import regression
from .utils import getEigenpairs

import itertools as it
import numpy as np
import numpy.linalg as la
import numpy.random as np_rnd
import os
import random as rnd
import sklearn.neighbors as skl_nb

# np.seterr(all='raise')


def lstsq_direction(xs, fs):
    """
    Computes the dominant direction by linear least squares approximation.

    :param xs: List of locations
    :param fs: List of function values
    """
    A = np.hstack([xs, np.ones(len(xs))[:, np.newaxis]])
    a = la.lstsq(A, fs)[0][:-1]
    return a / la.norm(a)


def _computeBootstrapEigenpairs(gradOuterSamples, nBoot):
    M = len(gradOuterSamples)
    bootEigVals = [None] * nBoot
    bootEigVecs = [None] * nBoot

    for iBoot in range(0, nBoot):
        replSamples = [rnd.choice(gradOuterSamples) for _ in range(M)]
        replC = np.sum(replSamples, axis=0) / float(M)
        replEigVals, replEigVecs = getEigenpairs(replC)
        bootEigVals[iBoot] = replEigVals
        bootEigVecs[iBoot] = replEigVecs

    return (np.array(bootEigVals), np.array(bootEigVecs))


def _computeBootstrapIntervals(bootEigVals, bootEigVecs, eigVecs):
    # Bootstrap interval for subspace errors
    numEigVals = bootEigVals.shape[1]
    nBoot = len(bootEigVecs)
    subspaceErrors = np.empty([numEigVals - 1, nBoot])
    for iDim in range(1, numEigVals):
        for iBootEigVecs in range(nBoot):
            replEigVecs = bootEigVecs[iBootEigVecs]
            subspaceErrors[iDim - 1, iBootEigVecs] = la.norm(
                np.matmul(eigVecs[:, 0:iDim].T, replEigVecs[:, iDim:]), 2)

    return (np.min(bootEigVals, 0), np.max(bootEigVals, 0), np.min(subspaceErrors, 1), np.max(subspaceErrors, 1), np.mean(subspaceErrors, 1))


def computeDataMisfitGradient(Gs, jacGs, bayInvPb):
    """
    Computes the gradient of the data misfit for given QoI evaluations and Jacobians.

    :param Gs: List of QoI evaluations
    :param jacGs: List of QoI Jacobians
    :param bayInvPb: Object representing a Bayesian inverse problem
    """
    M = len(Gs)
    if M != len(jacGs):
        raise Exception(
            'Different sizes of forward evaluations and their Jacobians.')

    gradOuterSamples = [None] * M

    i = 0
    for G, jacG in zip(Gs, jacGs):
        jacF = bayInvPb.misfitGradientG(G, jacG)

        gradOuterSamples[i] = np.outer(jacF.T, jacF)
        i += 1

    return np.array(gradOuterSamples, dtype=float)


def computeQoIAndJacobian(absProblem, id, params):
    """
    Computes the QoI and corresponding Jacobian of a given Bayesian inverse problem

    :param absProblem: Object representing an abstract problem able to instantiate a concrete (parameter dependent) problem
    :param params: Parameters
    :param string dir: Directory to store results in
    """
    pb = absProblem.instantiate(params, id)

    # Get QoIs with the solution of the forward problem
    qoi = pb.getQoI()

    # Get Jacobian of the forward map
    jac = pb.getJacobian()

    return qoi, jac


def computeActiveSubspace(gradOuterSamples, nBoot):
    """
    Computes the active subspace, i.e. it returns eigenvalues, eigenvectors and all bootstrap quantities.

    :param gradOuterSamples: List of Jacobian outer products
    :param integer nBoot: Number of re-computations for bootstrapping
    """
    C = assembleC(gradOuterSamples)

    eigVals, eigVecs = getEigenpairs(C)

    bootEigVals, bootEigVecs = _computeBootstrapEigenpairs(
        gradOuterSamples, nBoot)
    bootEigVals = np.insert(bootEigVals, 0, eigVals, axis=0)

    minEigVals, maxEigVals, minSubspaceErrors, maxSubspaceErrors, meanSubspaceErrors = _computeBootstrapIntervals(
        bootEigVals, bootEigVecs, eigVecs)

    return (eigVals, eigVecs, minEigVals, maxEigVals, minSubspaceErrors, maxSubspaceErrors, meanSubspaceErrors)


def computeActiveSubspaceFromSamples(Gs, jacGs, bayInvPb, nBoot, scaleMatrices=None):
    """
    Runs the active subspace method with already computed samples.

    :param Gs: List with forward run samples
    :param jacGs: List with gradient samples
    :param integer nBoot: Number of re-computations for bootstrapping
    """
    gradOuterSamples = computeDataMisfitGradient(Gs, jacGs, bayInvPb)

    if scaleMatrices is not None:
        assert len(gradOuterSamples) == len(scaleMatrices)
        gradOuterSamples = [np.dot(np.dot(scaleMatrices[i].T, gradOuterSamples[i]), scaleMatrices[i])
                            for i in range(len(gradOuterSamples))]

    result = computeActiveSubspace(gradOuterSamples, nBoot)

    return result


def assembleC(gradOuterSamples):
    return np.sum(gradOuterSamples, axis=0) / float(len(gradOuterSamples))


def assembleCFromSamples(Gs, jacGs, bayInvPb):
    return assembleC(computeDataMisfitGradient(Gs, jacGs, bayInvPb))


def computeIntrinsicDimensionFromSamples(Gs, jacGs, bayInvPb):
    C = assembleC(computeDataMisfitGradient(Gs, jacGs, bayInvPb))
    return np.trace(C) / la.norm(C, 2)


def response_surface(xs, fs, W1, poly_order=2):
    """
    Constructs a response surface on a specified subspace.

    1. Computes y_i = W1^T*x_i samples.
    2. Finds polynomial regression fit g such that g(y_i)=f_i

    :param xs: List with points in the original space
    :param fs: List with function evaluations
    :param W1: Matrix whose range specifies the subspace (active variable)
    :param poly_order: Polynomial order for regression
    """
    ys = np.dot(xs, W1)
    return regression.polynomial_fit(ys, fs, order=poly_order)


def marginalPriorY(prior_samples, W1, kde_bandwidth=0.05, kde_kernel='gaussian'):
    """
    Computes the marginal prior distribution on the active variable y.

    :param prior_samples: Prior samples
    :param W1: Matrix whose range specifies the subspace (active variable)
    :param kde_bandwidth: Bandwidth parameter for kernel density estimation
    :param kde_kernel: Kernel used for kernel density estimation (can be any kernel allowed by SciKit Learn)
    """
    n = np.shape(W1)[1]
    ys = np.dot(prior_samples, W1)

    kde = skl_nb.KernelDensity(
        bandwidth=kde_bandwidth, kernel=kde_kernel).fit(ys)

    return lambda y: np.exp(kde.score_samples(y[np.newaxis, :] if len(np.shape(y)) <= 1 else y))


def activeToOriginalMCMC(activeSamples, W1, W2, prior, proposal_sampler, z1, stepsPerActiveSample, burnIn=50000, maxlagInact=None, nPlotAccptRate=50):
    """
    Runs MCMC on the inactive variables (conditioned on active variable) to construct samples in the original space.

    :param activeSamples: List with active samples
    :param W1: Matrix specifying active subspace
    :param W2: Matrix specifying inactive subspace
    :param prior: Function representing the prior on the original space
    :param proposal_sampler: Function producing MCMC proposals
    :param z1: Start value for every MCMC run
    :param integer stepsPerActiveSample: Number of MCMC steps per active sample
    :param integer burnIn: Number of samples regarded as burn-in
    :param integer maxlagInact: Number of autocorrelations taken into account for computing effective sample size of inactive samples
    :param nPlotAccptRate: Distance between two outputs of the acceptance rate
    """
    inactiveSamplesList = [mcmc.mh_mcmc(lambda z: prior(np.dot(activeSample, W1.T) + np.dot(z, W2.T)),
                                        proposal_sampler, z1, stepsPerActiveSample, nPlotAccptRate) for activeSample in activeSamples]

    effInactiveSamplesList = [mcmc.pickEffSamples(inactiveSamples, burnIn, maxlag=maxlagInact)
                              for inactiveSamples in inactiveSamplesList]

    lens = list(map(len, effInactiveSamplesList))
    minLen = np.min(lens)
    effInactiveSamplesList = [samples[np_rnd.choice(
        len(samples), minLen)] for samples in effInactiveSamplesList]

    return np.array([[(np.dot(activeSample, W1.T) + np.dot(inactiveSample, W2.T))
                      for inactiveSample in effInactiveSamples]
                     for activeSample, effInactiveSamples in zip(activeSamples, effInactiveSamplesList)])


# Compute approximate (reduced) misfit at y_ in direction W1
def averaged_misfit(W1, W2, misfit, cond_prior_sampler, M):
    """
    Returns a function computing the conditional expectation of a misfit conditioned on the active sample put in.

    :param W1: Matrix specifying active subspace
    :param W2: Matrix specifying inactive subspace
    :param misfit: Data misfit function on the original space
    :param prior_cond_sampler: Function sampling an inactive sample conditioned on active sample put in
    :param integer M: Number of Monte Carlo summands for approximating the integral
    """
    return lambda y: np.average([misfit(np.dot(W1, y) +
                                                     np.dot(W2, cond_prior_sampler(y))) for _ in range(M)])


def as_mcmc(W1, W2, reduced_misfit, proposal_sampler, priorY, y1, steps=10**3, nPlotAccptRate=50):
    """
    Runs MCMC in the active subspace and returns also original samples if prior_cond_sampler is not None.

    :param W1: Matrix specifying active subspace
    :param W2: Matrix specifying inactive subspace
    :param reduced_misfit: Low-dimensional approximation of the data misfit function
    :param proposal_sampler: Function producing MCMC proposals
    :param priorY: Function representing the prior on the active variable
    :param prior_cond_sampler: Function sampling an inactive sample conditioned on active sample put in
    :type prior_cond_sampler: Function in one variable or None
    :param y1: Starting point for MCMC
    :param integer steps: Number of MCMC steps
    :param integer nCondInactSamples: Number of conditional inactive samples per active samples (only active if prior_cond_sampler not None)
    :param nPlotAccptRate: Distance between two outputs of the acceptance rate
    """
    ySamples = np.empty((steps, len(y1)))
    yk = y1
    ySamples[0, :] = yk
    gyk = reduced_misfit(yk)
    k = 1
    accptd = 0

    while k < steps:
        y_ = proposal_sampler(yk)
        gy_ = reduced_misfit(y_)
        priorY_y_ = priorY(y_)

        accpt_ratio = np.min(
            [1, np.exp(gyk - gy_) * priorY_y_ / priorY([yk])]) if priorY_y_ else 0

        if accpt_ratio >= np_rnd.uniform():
            yk = y_
            gyk = gy_
            accptd += 1

        ySamples[k, :] = yk

        if (k + 1) % nPlotAccptRate == 1:
            print("State %i: %s" % (k, repr(yk)))
            print("Acceptance rate at step %i: %s" % (
                k, "{0: .3}".format(accptd / float(k) * 100)))

        k += 1

    return ySamples


def as_mcmc_with_averaged_misfit(W1, W2, misfit, proposal_sampler, priorY, prior_cond_sampler, y1, M=10, steps=10**3, nPlotAccptRate=50):
    return as_mcmc(W1, W2, averaged_misfit(W1, W2, misfit, prior_cond_sampler, M), proposal_sampler, priorY, y1, steps, nPlotAccptRate)


def as_mcmc_with_response_surface(resp_surface, W1, W2, proposal_sampler, priorY, y1, steps=10**3, nPlotAccptRate=50):
    return as_mcmc(W1, W2, resp_surface, proposal_sampler, priorY, y1, steps, nPlotAccptRate)


def combineTwoActiveSubspaces(W1, W2):
    """
    Combines two active subspaces into one.

    For details, see Cortesi, et.al., 2017, Forward and backward uncertainty quantification with active subspaces: application to hypersonic flows around a cylinder.

    :param W1: Matrix specifying the first active subspace
    :param W2: Matrix specifying the second active subspace
    """
    W = np.hstack((W1, W2))
    U, R = la.qr(W)
    v, s, __ = la.svd(W)

    assert len(s) == np.shape(W)[1]
    V = v[:, len(s):]

    # Check for orthogonality
    A = np.hstack((U, V))
    assert np.equal(np.shape(A)[0], np.shape(A)[1])
    assert np.allclose(np.dot(A.T, A), np.eye(len(A)))
    # -------------------------------------------------

    return U, R, V


def compute_activity_scores(eig_vals, eig_vecs):
    return np.dot(eig_vecs*eig_vecs, eig_vals)
