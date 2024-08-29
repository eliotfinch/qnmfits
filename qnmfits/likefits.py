import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy import linalg as LA

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Class to load QNM frequencies and mixing coefficients
from .qnm import qnm
qnm = qnm()

# the mismatch function is defined in the other file
from .qnmfits import mismatch


def GPkernel(t1, t2, sigma=1.0e-3, timescale=1., jitter=1.0e-6):
    """
    Computes the square exponential Gaussian process kernel.
    
    Parameters
    ----------
    t1: array, shape = (m,)
        Array of sample times
    t2: array, shape = (n,)
        Array of sample times

    Kernel Parameters
    -----------------
    sigma: float
        amplitude scale for the NR uncertainty
    timescale: float
        time scale of correlated errors
    jitter: float
        diag term for numerical stability
        (should usually choose this to be a few orders of magnitude
        smaller than sigma)
    
        
    Returns
    -------
    K: array, shape = (n, m)
        The covariance matrix 
    """
    K = sigma**2 * np.exp(-0.5*((t1[:,np.newaxis]-t2[np.newaxis,:])/timescale)**2)

    if t1.shape == t2.shape:
        K += jitter**2 * np.eye(t1.shape[0])

    return K


def likelihood_fit(times, data, modes, Mf, chif, t0, GPkernel, GPkernel_kwargs={}, t0_method='geq', T=100):
    """
    Perform a least-squares fit to some data using a ringdown model.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data : array_like
        The data to be fitted by the ringdown model.
        
    modes : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to include in 
        the ringdown model. For regular (positive real part) modes use 
        sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
        
    Mf : float
        The remnant black hole mass, which along with chif determines the QNM
        frequencies.
        
    chif : float
        The magnitude of the remnant black hole spin.
        
    t0 : float
        The start time of the ringdown model.

    GPkernel : function
        The Gaussian process kernel to use in the likelihood fit. 
        Function should take two vectors of times, $t1\in\mathbb{R}^m$ and $t2\in\mathbb{R}^n$, 
        and kwargs, and return a covariance matrix of shape $m\times n$.

    GPkernel_kwargs : dict
        Keyword arguments to pass to the kernel function.
        
    t0_method : str, optional
        A requested ringdown start time will in general lie between times on
        the default time array (the same is true for the end time of the
        analysis). There are different approaches to deal with this, which can
        be specified here.
        
        Options are:
            
            - 'geq'
                Take data at times greater than or equal to t0. Note that
                we still treat the ringdown start time as occuring at t0,
                so the best fit coefficients are defined with respect to 
                t0.

            - 'closest'
                Identify the data point occuring at a time closest to t0, 
                and take times from there.
                
        The default is 'geq'.
        
    T : float, optional
        The duration of the data to analyse, such that the end time is t0 + T. 
        The default is 100.

    Returns
    -------
    best_fit : dict
        A dictionary of useful information related to the fit. Keys include:
            
            - 'residual' : float
                The residual from the fit.
            - 'mismatch' : float
                The mismatch between the best-fit waveform and the data.
            - 'C' : ndarray
                The best-fit complex amplitudes. There is a complex amplitude 
                for each ringdown mode.
            - 'data' : ndarray
                The (masked) data used in the fit.
            - 'model': ndarray
                The best-fit model waveform.
            - 'model_times' : ndarray
                The times at which the model is evaluated.
            - 't0' : float
                The ringdown start time used in the fit.
            - 'modes' : ndarray
                The ringdown modes used in the fit.
            - 'mode_labels' : list
                Labels for each of the ringdown modes (used for plotting).
            - 'frequencies' : ndarray
                The values of the complex frequencies for all the ringdown 
                modes.
            - 'Fisher_matrix_post' : ndarray 
                The Fisher matrix for the posterior distribution of the complex
                The shape of the Fisher matrix is (n_modes, n_modes).
            - 'FM_post_inv' : ndarray
                The inverse of the Fisher matrix.
            - 'FM_post_col_names' : list
                list of strings, suitable for use in matplotlib axis labels, for the 
                rows and cols of the Fisher matrix.
            - 'GPkernel_func_name' : str
                The name of the Gaussian process kernel function used in the fit.
                Saves GPkernel.__str__(). This is itended as a reminder of which GP 
                covariance function was used in the construction of the posterior.
            - 'GPkernel_kwargs' : dict 
                The dictionary of keyword arguments that were passed to the GPkernel 
                function. This is itended as a reminder of which GP covariance function 
                was used in the construction of the posterior.
    """
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0) & (times<t0+T)
        
        times = times[data_mask]
        data = data[data_mask]
        
    elif t0_method == 'closest':
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        
        times = times[start_index:end_index]
        data = data[start_index:end_index]
        
    else:
        print("""Requested t0_method is not valid. Please choose between 'geq'
              and 'closest'""")
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(modes, chif, Mf))

    # Construct coefficient matrix and solve
    # --------------------------------------
    
    # Construct the coefficient matrix
    a = np.array([
        np.exp(-1j*frequencies[i]*(times-t0)) for i in range(len(frequencies))
        ]).T
    
    # Gaussian process uncertainty kernel
    # -----------------------------------
    
    # Construct the covariance matrix
    dt = times[1] - times[0]
    assert np.allclose(np.diff(times)-dt, 0), "This only works with regularly spaced data" 
    cov = GPkernel(times, times, **GPkernel_kwargs) / dt**2

    # Use the Cholesky decomposition of the covariance matrix
    L = LA.cholesky(cov, lower=True)
    Linv_data = LA.cho_solve((L, True), data)
    Linv_a = LA.cho_solve((L, True), a)

    # Calculate the Fisher matrix for the posterior distribution
    FM_post_col_names = [ part + '(A_{' + str(mode) + '})$' 
                         for mode in modes 
                         for part in ['$\mathrm{Re}', '$\mathrm{Im}']]
    n_par = len(FM_post_col_names)
    FM_post = np.zeros((n_par, n_par))
    for i, mode in enumerate(modes):
        for j, part in enumerate(['Re', 'Im']):
            for i_, mode_ in enumerate(modes):
                for j_, part_ in enumerate(['Re', 'Im']):
                    A = np.real(Linv_a[:,i] ) if part =='Re' else np.imag(Linv_a[:,i] )
                    B = np.real(Linv_a[:,i_]) if part_=='Re' else np.imag(Linv_a[:,i_])
                    FM_post[2*i+j,2*i_+j_] = np.dot(A, B)

    vals, vecs = np.linalg.eigh(FM_post)
    if np.any(vals<0.):
        print('Warning: negative eigenvalues in Fisher matrix')
    FM_post_inv = np.einsum('ia,a,ja->ij', vecs, 1/np.abs(vals), vecs)
    
    # Solve for the complex amplitudes, C. Also returns the sum of residuals,
    # the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(Linv_a, Linv_data, rcond=None)
    
    # Evaluate the maximum likelihood model
    model = np.einsum('ij,j->i', a, C)
    
    # Calculate the mismatch for the fit
    mm = mismatch(times, model, data)
    
    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in modes]

    # Calculate the log likelihood of the best fitting model
    residuals = data - model
    logL = np.real(-0.5*np.dot(LA.cho_solve((L, True), residuals), residuals.conj()))
    
    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'mismatch': mm,
        'C': C,
        'data': data,
        'model': model,
        'model_times': times,
        't0': t0,
        'modes': modes,
        'mode_labels': labels,
        'frequencies': frequencies,
        'max_logL': logL,
        'Fisher_matrix_post' : FM_post ,
        'FM_post_inv' : FM_post_inv ,
        'FM_post_col_names' : FM_post_col_names ,
        'GPkernel_func_name' : GPkernel.__str__(),
        'GPkernel_kwargs' : GPkernel_kwargs
        }
    
    # Return the output dictionary
    return best_fit
