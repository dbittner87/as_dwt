import itertools as it
import numpy as np
import numpy.fft as fft
import numpy.random as rnd


def mh_mcmc(dens, proposal_sampler, x1, steps, nPlotAccptRate=50):
    """
    Runs standard Metropolis-Hastings algorithm for MCMC.

    :param dens: Function representing the density to generate samples from
    :param proposal_sampler: Function producing MCMC proposals
    :param x1: Starting point for MCMC
    :param integer steps: Number of MCMC steps
    :param nPlotAccptRate: Distance between two outputs of the acceptance rate
    """
    samples = np.empty((steps, len(x1)))
    xk = x1
    samples[0, :] = xk
    dens_xk = dens(xk)
    k = 1
    accptd = 0

    while k < steps:
        x_ = proposal_sampler(xk)
        dens_x_ = dens(x_)
        # print x_
        # print "MCMC: %f" % dens_x_

        accpt_ratio = np.min([1, dens_x_ / dens_xk]) \
            if dens_xk > 0 else \
            (1 if dens_x_ > 0 else 0)

        if accpt_ratio >= rnd.uniform():
            xk = x_
            dens_xk = dens_x_
            accptd += 1

        samples[k, :] = xk

        if (k + 1) % nPlotAccptRate == 1:
            print("State %i: %s" % (k, xk))
            print("Acceptence rate at step %i: %s" % (
                k, "{0:.3}".format(accptd / float(k) * 100)))

        k += 1

    return samples


def bi_mh_mcmc(misfit, prior, proposal_sampler, x1, steps, nPlotAccptRate=50):
    """
    Runs Metropolis-Hastings for Bayesian inversion.

    :param misfit: Function representing the data misfit function
    :param prior: Function representing the prior
    :param proposal_sampler: Function producing MCMC proposals
    :param x1: Starting point for MCMC
    :param integer steps: Number of MCMC steps
    :param nPlotAccptRate: Distance between two outputs of the acceptance rate
    """
    samples = np.empty((steps, len(x1)))
    xk = x1
    samples[0, :] = xk
    gxk = misfit(xk)
    k = 1
    accptd = 0

    while k < steps:
        x_ = proposal_sampler(xk)
        gx_ = misfit(x_)
        prior_x_ = prior(x_)

        accpt_rat = np.min(
            [1, np.exp(gxk - gx_) * prior_x_ / prior(xk)]) if prior_x_ > 0 else 0

        if accpt_rat >= rnd.uniform():
            xk = x_
            gxk = gx_
            accptd = accptd + 1

        samples[k, :] = xk

        if (k + 1) % nPlotAccptRate == 1:
            print("State " + repr(k) + ": " + repr(xk))
            print("Acceptence rate at step " + repr(k) + ": " + \
                "{0:.3}".format(accptd / float(k) * 100))

        k = k + 1

    return samples


def _autocorr(samples):
    N = len(samples)
    m = np.mean(samples)

    nfft = 2**int(np.ceil(np.log2(N)))
    freq = fft.fft(samples - m, n=nfft)
    acf = np.real(fft.ifft(freq * np.conj(freq))[:N]) / 4 * nfft

    return acf / acf.flat[0]


def _autocorrTime(samples, maxlag):
    acf = _autocorr(samples)

    if maxlag is None:
        acf = np.array(list(it.takewhile(lambda x: x > 0., acf)))
        maxlag = len(acf)
    # print "Maxlag: ", len(acf)

    return 1 + 2 * np.sum(acf), acf, maxlag


def stats(samples, burnIn, maxlag=None):
    """
    Computes statistics for MCMC samples.

    The statistics consist of the minimum effective sample size (ESS), the ESSs, the autocorrelation times and the autocorrelations for every component

    :param samples: List of MCMC samples
    :param integer burnIn: Number of samples regarded as burn-in
    :param integer maxlag: Number of autocorrelations taken into account for computing ESSs
    """
    samples = samples[burnIn:]
    N, dims = np.shape(samples)
    ess = np.empty(dims)
    autoCorrTimes = np.empty(dims)
    acfs = [None] * dims
    maxlags = np.empty(dims)

    for i in range(dims):
        autoCorrTime, acf, maxlags[i] = _autocorrTime(samples[:, i], maxlag)
        ess[i] = N / autoCorrTime
        autoCorrTimes[i] = autoCorrTime
        acfs[i] = acf

    if maxlag is None:
        # Cut every acf on the minimal size of all acf's
        # l = np.min(map(len, acfs))
        maxlag = int(np.min(maxlags))

    acfs = [acf[:maxlag] for acf in acfs]

    return np.min(ess), ess, autoCorrTimes, np.array(acfs).T


def pickEffSamples(samples, burnIn, maxlag=None):
    """
    Picks effective samples out of a set of sample according to the effective sample size (ESS).

    :param samples: List of MCMC samples
    :param integer burnIn: Number of samples regarded as burn-in
    :param integer maxlag: Number of autocorrelations taken into account for computing ESSs
    """
    minEss = stats(samples, burnIn, maxlag)[0]
    return samples[burnIn::(len(samples)-burnIn) / int(minEss)]
