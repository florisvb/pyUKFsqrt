import numpy as np
import numpy.linalg as linalg
import scipy.linalg

from ukf_sqrt.utils import cholupdate

def ukf_sqrt(y, x0, f, h, Q, R, u, alpha=0.001, beta=2):

    # %-----------------------------------------------------------------------
    # %Copyright (C) Floris van Breugel, 2021.
    # %  
    # %florisvb@gmail.com
    # %
    # %This function was originally written for MATLAB by Nathan Powell
    # %
    # %Released under the GNU GPL license, Version 3
    # %
    # %pyUKFsqrt is free software: you can redistribute it and/or modify it
    # %under the terms of the GNU General Public License as published
    # %by the Free Software Foundation, either version 3 of the License, or
    # %(at your option) any later version.
    # %    
    # %pyUKFsqrt is distributed in the hope that it will be useful, but WITHOUT
    # %ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    # %FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
    # %License for more details.
    # %
    # %You should have received a copy of the GNU General Public
    # %License along with pyUKFsqrt.  If not, see <http://www.gnu.org/licenses/>.
    # %
    # %------------------------------------------------------------------------

    '''
    Unscented Kalman Filter, square root implementation. 

    Inputs
    ======
    y  --  measurements, np.matrix [m, N]
    x0 --  initial state, np.matrix [k, 1]
    f  --  function describing process dynamics, with inputs of (x, u, w), returns state estimate xhat
    h  --  function describing observation dynamics, with inputs of (x, u, w), returns measurements
    Q  --  Process covariance as a function of time, np.matrix [k, k, N]
    R  --  Measurement covariance as a function of time, np.matrix [m, m, N]
    u  --  Control inputs, np.matrix [k, N]

    m: number of measurements
    k: number of states

    Returns
    =======
    x  --  Full state estimate, np.matrix [k, N]
    P  --  Covariance matrices, np.matrix [k, k, N]
    s  --  Some other covariance metric.. not quite sure...

    '''

    N = y.shape[1]

    nx = x0.shape[0]
    ny = y.shape[0]
    nq = Q.shape[0]
    nr = R.shape[0]

    a = alpha
    b = beta
    L = nx + nq + nr
    l = a**2*L - L
    g = np.sqrt(L + l)

    Wm = np.hstack(([[l/(L + l)]],  1/(2*(L + l))*np.ones([1, 2*L]))) # Weights for means
    Wm = np.matrix(Wm)
    Wc = np.hstack(([[(l/(L + l) + (1 - a**2 + b))]], 1/(2*(L + l))*np.ones([1, 2*L]) )) # Weights for covariances
    Wc = np.matrix(Wc)

    if Wc[0,0] >= 0:
        sgnW0 = 1
    else:
        sgnW0 = -1

    ix = np.arange(0, nx)
    iy = np.arange(0, ny)
    iq = np.arange(nx, (nx+nq))
    ir = np.arange((nx+nq), (nx+nq+nr))

    Sa = np.zeros([L,L])
    Sa[np.ix_(iq, iq)] = linalg.cholesky(Q[:,:,0])#.T
    Sa[np.ix_(ir, ir)] = linalg.cholesky(R[:,:,0])#.T

    Y = np.zeros([ny, 2*L+1]) # Measurements from propagated sigma points
    x = np.zeros([nx,N]) # Unscented state estimate
    P = np.zeros([nx,nx,N]) # Unscented estimated state covariance
    ex = np.zeros([nx, 2*L+1])
    ey = np.zeros([ny, 2*L+1])

    x[:,0:1] = x0
    P[:,:,0] = 1*np.eye(nx) #np.diag(P0)
    S = linalg.cholesky(P[:,:,0])#.T

    for i in range(1, N):
        Sa[np.ix_(ix, ix)] = S

        # Only do this if R actually is time dependent
        Sa[np.ix_(iq, iq)] = linalg.cholesky(Q[:,:,i]) #.T #chol(Q(:,:,i));
        Sa[np.ix_(ir, ir)] = linalg.cholesky(R[:,:,i]) #.T #chol(R(:,:,i));
        
        #Sa = nearestPD(Sa)

        xa = np.vstack([x[:,i-1:i], np.zeros([nq,1]), np.zeros([nr,1])])
        gsa = np.hstack((g*Sa.T, -g*Sa.T)) + xa*np.ones([1, 2*L])
        X = np.hstack([xa, gsa])

        # Propagate sigma points
        for j in range(0, 2*L+1):
            X[np.ix_(ix, [j])] = f(X[np.ix_(ix, [j])], 
                                   u[:,i-1:i], 
                                   X[np.ix_(iq, [j])])

            Y[:, j:j+1] = h(X[np.ix_(ix, [j])], 
                            u[:,i-1:i], 
                            X[np.ix_(ir, [j])])

            
        # Average propagated sigma points
        x[:,i:i+1] = X[np.ix_(ix, np.arange(0, X.shape[1]))]*Wm.T
        yf = Y*Wm.T
        
        # Calculate new covariances
        Pxy = np.zeros([nx,ny])
        for j in range(0, (2*L)+1):
            ex[:,j:j+1] = np.sqrt(np.abs(Wc[0,j]))*(X[np.ix_(ix, [j])] - x[:,i:i+1])
            ey[:,j:j+1] = np.sqrt(np.abs(Wc[0,j]))*(Y[:,j:j+1] - yf)
            Pxy = Pxy + Wc[0,j]*(X[np.ix_(ix, [j])] - x[:,i:i+1])*(Y[:,j:j+1] - yf).T

        qr_Q, qr_R = scipy.linalg.qr( (ex[:, 1:].T) )
        S = cholupdate(qr_R[np.ix_(ix, ix)], ex[:, 0], sgnW0)

        qr_Q, qr_R = scipy.linalg.qr( ey[:, 1:].T )
        Syy = cholupdate(qr_R[np.ix_(iy, iy)], ey[:, 0], sgnW0)
        Syy = Syy

        # Update unscented estimate
        K = Pxy*np.linalg.pinv(Syy.T*Syy)
        x[:,i:i+1] = x[:,i:i+1] + K*(y[:,i:i+1] - h(x[:,i:i+1], u[:,i:i+1], np.zeros([nr,1])));
        U = K*Syy.T
        for j in range(ny):
            S = cholupdate(S, np.ravel(U[:,j]), -1)
        #S = nearestPD(S)

        P[:,:,i] = S.T*S
        #P[:,:,i] = nearestPD(P[:,:,i])
        
    s = np.zeros([nx,y.shape[1]]);
    for i in range(nx):
        s[i,:] = np.sqrt( P[i,i,:].squeeze() )
        
    return x, P, s