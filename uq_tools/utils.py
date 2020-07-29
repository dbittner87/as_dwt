import math
import multiprocessing as mp
import numpy as np
import numpy.linalg as la
import os
import sys


def getEigenpairs(C):
    """Computes eigenpairs of a matrix in decreasing order of eigenvalues."""
    eigVals, eigVecs = la.eigh(C)

    idx = eigVals.argsort()[::-1]
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:, idx]

    return (eigVals, eigVecs)


def volNBall(r, dim):
    """
    Computes the volume of an n-dimensional ball.

    :param float r: Radius
    :param integer dim: Dimension
    """
    return math.pi**(0.5 * dim) * (r**dim) / (math.gamma(0.5 * dim + 1))


def kOutOfn(n, k):
    """Generates lists combinatorically with k 'True's out of n."""
    if n == 0:
        return []
    elif k == 0:
        return [[False] * n]
    elif n == k:
        return [[True] * k]
    else:
        return [[True] + li for li in kOutOfn(n - 1, k - 1)] + [[False] + li for li in kOutOfn(n - 1, k)]


# Nash-Sutcliffe efficiency coefficient (NSE)
def nse(modelOutput, data):
    data_mean = np.mean(data)
    return 1 - la.norm(modelOutput-data, 2)**2 / la.norm(data-data_mean, 2)**2


def grad_nse(jacModelOutput, modelOutput, data):
    data_mean = np.mean(data)
    return -2./la.norm(data-data_mean, 2)**2 * np.dot(jacModelOutput.T, modelOutput-data)


def save_active_subspace(eigVals, eigVecs, minEigVals, maxEigVals, minSubspaceErrors, maxSubspaceErrors, meanSubspaceErrors, dir, prefix=None):
    prefix = prefix + '_' if prefix is not None else ''

    np.savetxt('%s/%sasm_eigVals.txt' % (dir, prefix), eigVals)
    np.savetxt('%s/%sasm_eigVecs.txt' % (dir, prefix), eigVecs)
    np.savetxt('%s/%sasm_minEigVals.txt' % (dir, prefix), minEigVals)
    np.savetxt('%s/%sasm_maxEigVals.txt' % (dir, prefix), maxEigVals)
    np.savetxt('%s/%sasm_minSubspaceErrors.txt' %
               (dir, prefix), minSubspaceErrors)
    np.savetxt('%s/%sasm_maxSubspaceErrors.txt' %
               (dir, prefix), maxSubspaceErrors)
    np.savetxt('%s/%sasm_meanSubspaceErrors.txt' %
               (dir, prefix), meanSubspaceErrors)


def construct_G_jacG_parallel(samples, script_name, dir, cpu_count=None):
    np.savetxt('%s/samples.txt' % dir, samples)

    cmds = ''
    for i in range(len(samples)):
        sample = samples[i]
        cmd = sys.executable + " " + script_name + " " + \
            str(i) + " " + ' '.join(str(p) for p in sample)
        cmds += cmd + ' > /dev/null\n'

    launcherFile = '%s/asmLauncher_runs.txt' % dir

    with open(launcherFile, 'w') as file:
        file.write(cmds)

    os.environ["LAUNCHER_JOB_FILE"] = launcherFile
    os.environ["LAUNCHER_DIR"] = os.environ["SW_DIR"] + "/launcher"
    os.environ["LAUNCHER_PPN"] = str(cpu_count if cpu_count is not None else mp.cpu_count() - 1)

    os.system("bash $LAUNCHER_DIR/paramrun")
