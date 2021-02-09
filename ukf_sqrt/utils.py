import numpy as np
import numpy.linalg as linalg
import scipy
import scipy.linalg
from numpy import linalg as la

def cholupdate(A2, X, weight):
    '''Calculate chol(A + w x x')

    Parameters
    ----------
    A2 : [n_dim, n_dim] array
        A = A2.T.dot(A2) for A positive definite, symmetric
    X : [n_dim] or [n_vec, n_dim] array
        vector(s) to be used for x.  If X has 2 dimensions, then each row will be
        added in turn.
    weight : float
        weight to be multiplied to each x x'. If negative, will use
        sign(weight) * sqrt(abs(weight)) instead of sqrt(weight).

    Returns
    -------
    A2 : [n_dim, n_dim array]
        cholesky decomposition of updated matrix

    Notes
    -----

    Code based on the following MATLAB snippet taken from Wikipedia on
    August 14, 2012::

        function [L] = cholupdate(L,x)
            p = length(x);
            x = x';
            for k=1:p
                r = sqrt(L(k,k)^2 + x(k)^2);
                c = r / L(k, k);
                s = x(k) / L(k, k);
                L(k, k) = r;
                L(k,k+1:p) = (L(k,k+1:p) + s*x(k+1:p)) / c;
                x(k+1:p) = c*x(k+1:p) - s*L(k, k+1:p);
            end
        end
    '''
    # make copies
    X = X.copy()
    A2 = A2.copy()

    # standardize input shape
    if len(X.shape) == 1:
        X = X[np.newaxis, :]
    n_vec, n_dim = X.shape

    # take sign of weight into account
    sign, weight = np.sign(weight), np.sqrt(np.abs(weight))
    X = weight * X

    for i in range(n_vec):
        x = X[i, :]
        for k in range(n_dim):
            r_squared = A2[k, k] ** 2 + sign * x[k] ** 2
            r = 0.0 if r_squared < 0 else np.sqrt(r_squared)
            c = r / A2[k, k]
            s = x[k] / A2[k, k]
            A2[k, k] = r
            A2[k, k + 1:] = (A2[k, k + 1:] + sign * s * x[k + 1:]) / c
            x[k + 1:] = c * x[k + 1:] - s * A2[k, k + 1:]
    return A2


# Not necessary, but keeping around just in case
def nearestPD(A):
    """Find the nearest positive-definite matrix to input
    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    
    nan and inf checks added.
    """
    A[np.isnan(A)] = 0
    A[np.isinf(A)] = 0
    
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

# Not necessary but keeping around just in case
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False