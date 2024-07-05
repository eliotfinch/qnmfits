import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.interpolate import interp1d

from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Class to load QNM frequencies and mixing coefficients
from .qnm import qnm
qnm = qnm()


def ringdown(time, start_time, complex_amplitudes, frequencies):
    r"""
    The base ringdown function, which has the form
    
    .. math:: 
        h = h_+ - ih_\times
        = \sum_{\ell m n} C_{\ell m n} e^{-i \omega_{\ell m n} (t - t_0)},
             
    where :math:`C_{\ell m n}` are complex amplitudes, 
    :math:`\omega_{\ell m n} = 2\pi f_{\ell m n} - \frac{i}{\tau_{\ell m n}}` 
    are complex frequencies, and :math:`t_0` is the start time of the 
    ringdown.
    
    If start_time is after the first element of the time array, the model is 
    zero-padded before that time. 
    
    The amplitudes should be given in the same order as the frequencies they
    correspond to.

    Parameters
    ----------
    time : array_like
        The times at which the model is evalulated.
        
    start_time : float
        The time at which the model begins. Should lie within the times array.
        
    complex_amplitudes : array_like
        The complex amplitudes of the modes.
        
    frequencies : array_like
        The complex frequencies of the modes. These should be ordered in the
        same order as the amplitudes.

    Returns
    -------
    h : ndarray
        The plus and cross components of the ringdown waveform, expressed as a
        complex number.
    """
    # Create an empty array to add the result to
    h = np.zeros(len(time), dtype=complex)
    
    # Mask so that we only consider times after the start time
    t_mask = time >= start_time

    # Shift the time so that the waveform starts at time zero, and mask times
    # after the start time
    time = (time - start_time)[t_mask]
        
    # Construct the waveform, summing over each mode
    h[t_mask] = np.sum([
        complex_amplitudes[n]*np.exp(-1j*frequencies[n]*time)
        for n in range(len(frequencies))], axis=0)
        
    return h


def mismatch(times, wf_1, wf_2):
    """
    Calculates the mismatch between two complex waveforms.

    Parameters
    ----------
    times : array_like
        The times at which the waveforms are evaluated.
        
    wf_1, wf_2 : array_like
        The two waveforms to calculate the mismatch between.
        
    RETURNS
    -------
    M : float
        The mismatch between the two waveforms.
    """
    numerator = np.real(np.trapz(wf_1 * np.conjugate(wf_2), x=times))
    
    denominator = np.sqrt(np.trapz(np.real(wf_1 * np.conjugate(wf_1)), x=times)
                         *np.trapz(np.real(wf_2 * np.conjugate(wf_2)), x=times))
    
    return 1 - (numerator/denominator)


def multimode_mismatch(times, wf_dict_1, wf_dict_2):
    """
    Calculates the multimode (sky-averaged) mismatch between two dictionaries 
    of spherical-harmonic waveform modes. 
    
    If the two dictionaries have a different set of keys, the sum is performed
    over the keys of wf_dict_1 (this may be the case, for example, if only a 
    subset of spherical-harmonic modes are modelled).

    Parameters
    ----------
    times : array_like
        The times at which the waveforms are evaluated.
        
    wf_dict_1, wf_dict_2 : dict
        The two dictionaries of waveform modes to calculate the mismatch 
        between.

    RETURNS
    -------
    M : float
        The mismatch between the two waveforms.
    """    
    keys = list(wf_dict_1.keys())
    
    numerator = np.real(sum([
        np.trapz(wf_dict_1[key] * np.conjugate(wf_dict_2[key]), x=times) 
        for key in keys]))
    
    wf_1_norm = sum([
        np.trapz(np.real(wf_dict_1[key] * np.conjugate(wf_dict_1[key])), x=times) 
        for key in keys])
    
    wf_2_norm = sum([
        np.trapz(np.real(wf_dict_2[key] * np.conjugate(wf_dict_2[key])), x=times) 
        for key in keys])
    
    denominator = np.sqrt(wf_1_norm*wf_2_norm)
    
    return 1 - (numerator/denominator)


def ringdown_fit(times, data, modes, Mf, chif, t0, t0_method='geq', T=100, delta = 0):
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
    
    delta : float or array_like (optional)
        Modify the frequencies used in the ringdown fit. Either a
        constant value to modify every overtone frequency identically, or 
        an array with different values for each overtone.  Default is 0 
        (no modification).

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

    # Checking input for delta
    # If delta is list of appropriate length, converts it to np.array
    if type(delta) is list and len(delta)==len(modes):
        delta=np.array(delta)
    
    # If delta is a float or array of appropriate length, sets delta factor
    if (type(delta) is np.array and len(delta)==len(modes)) or type(delta) is float:
        delta_factor= delta+1
    
    else:
        print("delta must be a float or an array with length len(modes)")


    # Multiply frequencies by delta_factor = delta + 1
    frequencies = delta_factor*np.array(qnm.omega_list(modes, chif, Mf))
        
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    # Construct the coefficient matrix
    a = np.array([
        np.exp(-1j*frequencies[i]*(times-t0)) for i in range(len(frequencies))
        ]).T

    # Solve for the complex amplitudes, C. Also returns the sum of residuals,
    # the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
    # Evaluate the model
    model = np.einsum('ij,j->i', a, C)
    
    # Calculate the mismatch for the fit
    mm = mismatch(times, model, data)
    
    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in modes]
    
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
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit


def dynamic_ringdown_fit(times, data, modes, Mf, chif, t0, t0_method='geq', 
                         T=100):
    """
    Perform a least-squares fit to some data using a ringdown model. The 
    remnant mass and spin can be arrays of length time, which allows the Kerr 
    spectrum to change with time.

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
        
    Mf : float or array_like
        The remnant black hole mass, which along with chif determines the QNM
        frequencies. This can be a float, so that mass doesn't change with
        time, or an array of the same length as times.
        
    chif : float or array_like
        The magnitude of the remnant black hole spin. As with Mf, this can be
        a float or an array.
        
    t0 : float
        The start time of the ringdown model.
        
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
                The best-fit complex amplitudes. There is a (time dependant)
                complex amplitude for each ringdown mode.
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
    """
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0) & (times<t0+T)
        
        times = times[data_mask]
        data = data[data_mask]
        
    elif t0_method == 'closest':
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        data_mask = np.arange(len(times))[start_index:end_index]
        
        times = times[data_mask]
        data = data[data_mask]
        
    else:
        print("""Requested t0_method is not valid. Please choose between 'geq'
              and 'closest'""")
    
    if type(Mf) in [float, np.float64]:
        Mf = np.full(len(times), Mf)
    else:
        Mf = Mf[data_mask]
        
    if type(chif) in [float, np.float64]:
        chif = np.full(len(times), chif)
    else:
        chif = chif[data_mask]
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(modes, chif, Mf))
        
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    # Construct the coefficient matrix
    a = np.exp(-1j*frequencies*(times-t0)).T

    # Solve for the complex amplitudes, C. Also returns the sum of
    # residuals, the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
    # Evaluate the model. This needs to be split up into the separate
    # spherical harmonic modes.
    model = np.einsum('ij,j->i', a, C)
    
    # Calculate the (sky-averaged) mismatch for the fit
    mm = mismatch(times, model, data)
    
    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in modes]
    
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
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit

    
def multimode_ringdown_fit(times, data_dict, modes, Mf, chif, t0, 
                           t0_method='geq', T=100, spherical_modes=None):
    """
    Perform a least-squares ringdown fit to data which has been decomposed 
    into spherical-harmonic modes.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data_dict : dict
        The data (decomposed into spherical-harmonic modes) to be fitted by 
        the ringdown model. This should have keys (l,m) and array_like data of
        length times.
        
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
        
    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical-harmonic modes 
        the analysis should be performed on. If None, all the modes contained 
        in data_dict are used. The default is None.

    Returns
    -------
    best_fit : dict
        A dictionary of useful information related to the fit. Keys include:
            
            - 'residual' : float
                The residual from the fit.
            - 'mismatch' : float
                The mismatch between the best-fit waveform and the data.
            - 'C' : ndarray
                The (shared) best-fit complex amplitudes. There is a complex
                amplitude for each ringdown mode.
            - 'weighted_C' : dict
                The complex amplitudes weighted by the mixing coefficients. 
                There is a dictionary entry for each spherical mode.
            - 'data' : dict
                The (masked) data used in the fit.
            - 'model': dict
                The best-fit model waveform. Keys correspond to the spherical
                modes.
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
    """
    # Use the requested spherical modes
    if spherical_modes is None:
        spherical_modes = list(data_dict.keys())
    
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0) & (times<t0+T)
        
        times = times[data_mask]
        data = np.concatenate(
            [data_dict[lm][data_mask] for lm in spherical_modes])
        data_dict_mask = {lm: data_dict[lm][data_mask] for lm in spherical_modes}
        
    elif t0_method == 'closest':
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        
        times = times[start_index:end_index]
        data = np.concatenate(
            [data_dict[lm][start_index:end_index] for lm in spherical_modes])
        data_dict_mask = {
            lm: data_dict[lm][start_index:end_index] for lm in spherical_modes}
        
    else:
        print("""Requested t0_method is not valid. Please choose between
              'geq' and 'closest'.""")
    
    data_dict = data_dict_mask
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(modes, chif, Mf))
    
    # Construct the coefficient matrix for use with NumPy's lstsq function. 
    
    # Mixing coefficients
    # -------------------
    
    # A list of lists for the mixing coefficient indices. The first list is
    # associated with the first lm mode. The second list is associated with
    # the second lm mode, and so on.
    # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
    #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
    indices_lists = [
        [lm_mode+mode for mode in modes] for lm_mode in spherical_modes]
    
    # Convert each tuple of indices in indices_lists to a mu value
    mu_lists = [qnm.mu_list(indices, chif) for indices in indices_lists]
        
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    # Construct the coefficient matrix
    a = np.concatenate([np.array([
        mu_lists[i][j]*np.exp(-1j*frequencies[j]*(times-t0)) 
        for j in range(len(frequencies))]).T 
        for i in range(len(spherical_modes))])

    # Solve for the complex amplitudes, C. Also returns the sum of
    # residuals, the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
    # Evaluate the model. This needs to be split up into the separate
    # spherical harmonic modes.
    model = np.einsum('ij,j->i', a, C)
    
    # Split up the result into the separate spherical harmonic modes, and
    # store to a dictionary. We also store the "weighted" complex amplitudes 
    # to a dictionary.
    model_dict = {}
    weighted_C = {}
    
    for i, lm in enumerate(spherical_modes):
        model_dict[lm] = model[i*len(times):(i+1)*len(times)]
        weighted_C[lm] = np.array(mu_lists[i])*C
    
    # Calculate the (sky-averaged) mismatch for the fit
    mm = multimode_mismatch(times, model_dict, data_dict)
    
    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in modes]
    
    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'mismatch': mm,
        'C': C,
        'weighted_C': weighted_C,
        'data': data_dict,
        'model': model_dict,
        'model_times': times,
        't0': t0,
        'modes': modes,
        'mode_labels': labels,
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit


def dynamic_multimode_ringdown_fit(times, data_dict, modes, Mf, chif, t0, 
                                   t0_method='geq', T=100, 
                                   spherical_modes=None):
    """
    Perform a least-squares ringdown fit to data which has been decomposed 
    into spherical-harmonic modes. The remnant mass and spin can be arrays of
    length time, which allows the Kerr spectrum to change with time.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data_dict : dict
        The data (decomposed into spherical-harmonic modes) to be fitted by 
        the ringdown model. This should have keys (l,m) and array_like data of
        length times.
        
    modes : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to include in 
        the ringdown model. For regular (positive real part) modes use 
        sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
        
    Mf : float or array_like
        The remnant black hole mass, which along with chif determines the QNM
        frequencies. This can be a float, so that mass doesn't change with
        time, or an array of the same length as times.
        
    chif : float or array_like
        The magnitude of the remnant black hole spin. As with Mf, this can be
        a float or an array.
        
    t0 : float
        The start time of the ringdown model.
        
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
        
    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical-harmonic modes 
        the analysis should be performed on. If None, all the modes contained 
        in data_dict are used. The default is None.

    Returns
    -------
    best_fit : dict
        A dictionary of useful information related to the fit. Keys include:
            
            - 'residual' : float
                The residual from the fit.
            - 'mismatch' : float
                The mismatch between the best-fit waveform and the data.
            - 'C' : ndarray
                The (shared) best-fit complex amplitudes. There is a (time 
                dependant) complex amplitude for each ringdown mode.
            - 'weighted_C' : dict
                The complex amplitudes weighted by the mixing coefficients. 
                There is a dictionary entry for each spherical mode.
            - 'data' : dict
                The (masked) data used in the fit.
            - 'model': dict
                The best-fit model waveform. Keys correspond to the spherical
                modes.
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
    """
    # Use the requested spherical modes
    if spherical_modes is None:
        spherical_modes = list(data_dict.keys())
    
    # Mask the data with the requested method
    if t0_method == 'geq':
        
        data_mask = (times>=t0) & (times<t0+T)
        
        times = times[data_mask]
        data = np.concatenate(
            [data_dict[lm][data_mask] for lm in spherical_modes])
        data_dict_mask = {lm: data_dict[lm][data_mask] for lm in spherical_modes}
        
    elif t0_method == 'closest':
        
        start_index = np.argmin((times-t0)**2)
        end_index = np.argmin((times-t0-T)**2)
        data_mask = np.arange(len(times))[start_index:end_index]
        
        times = times[data_mask]
        data = np.concatenate(
            [data_dict[lm][data_mask] for lm in spherical_modes])
        data_dict_mask = {
            lm: data_dict[lm][data_mask] for lm in spherical_modes}
        
    else:
        print("""Requested t0_method is not valid. Please choose between
              'geq' and 'closest'.""")
    
    data_dict = data_dict_mask

    Mf = Mf[data_mask]
    if type(chif) in [float, np.float64]:
        chif = np.full(len(times), chif)
    else:
        chif = chif[data_mask]
    
    # Frequencies
    # -----------
    
    frequencies = np.array(qnm.omega_list(modes, chif, Mf)).T
    
    # We stack as many frequency arrays on top of each other as we have
    # spherical_modes
    frequencies = np.vstack(len(spherical_modes)*[frequencies])
        
    # Construct the coefficient matrix for use with NumPy's lstsq function.
    
    # Mixing coefficients
    # -------------------
    
    # A list of lists for the mixing coefficient indices. The first list is
    # associated with the first lm mode. The second list is associated with 
    # the second lm mode, and so on.
    # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
    #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
    indices_lists = [
        [lm_mode+mode for mode in modes] for lm_mode in spherical_modes]
    
    # Convert each tuple of indices in indices_lists to an array of mu values
    mu_lists = [qnm.mu_list(indices, chif) for indices in indices_lists]
    
    # I = len(spherical_modes)
    # J = len(modes)
    # K = len(times)
    
    # At this point, mu_lists has a shape (I, J, K). We want to reshape it 
    # into a 2D array of shape (I*K, J), such that the first K rows correspond 
    # to the first lm mode.
    
    # Flatten to make reshaping easier
    mu_lists = np.array([item for sublist in mu_lists for item in sublist]).T
    
    # The above flattens the array into a 2D array of shape (K, I*J). So, the
    # separate lm mode arrays are stacked horizontally, and not in the desired 
    # vertical way.
    
    # Reshape
    mu_lists = np.vstack([
        mu_lists[:,i*len(modes):(i+1)*len(modes)] 
        for i in range(len(spherical_modes))
        ])
    
    # The above reshaping converts the shape into the desired (I*K, J)
    
    # Construct coefficient matrix and solve
    # --------------------------------------
    
    stacked_times = np.vstack(len(spherical_modes)*[times[:,None]])
    a = mu_lists*np.exp(-1j*frequencies*(stacked_times-t0))

    # Solve for the complex amplitudes, C. Also returns the sum of
    # residuals, the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
    # Evaluate the model. This needs to be split up into the separate
    # spherical harmonic modes.
    model = np.einsum('ij,j->i', a, C)
    
    # Evaluate the weighted coefficients (which are now time dependant).
    # These also need to be split up into the separate spherical harmonic
    # modes.
    weighted_C = mu_lists*C
    
    # Split up the result into the separate spherical harmonic modes, and
    # store to a dictionary. We also store the "weighted" complex amplitudes 
    # to a dictionary.
    model_dict = {}
    weighted_C_dict = {}
    
    for i, lm in enumerate(spherical_modes):
        model_dict[lm] = model[i*len(times):(i+1)*len(times)]
        weighted_C_dict[lm] = weighted_C[i*len(times):(i+1)*len(times)]
    
    # Calculate the (sky-averaged) mismatch for the fit
    mm = multimode_mismatch(times, model_dict, data_dict)
    
    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in modes]
    
    # Store all useful information to a output dictionary
    best_fit = {
        'residual': res,
        'mismatch': mm,
        'C': C,
        'weighted_C': weighted_C_dict,
        'data': data_dict,
        'model': model_dict,
        'model_times': times,
        't0': t0,
        'modes': modes,
        'mode_labels': labels,
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit


def plot_ringdown(times, data, xlim=[-50,100], best_fit=None, 
                  spherical_mode=None, log=False, outfile=None, fig_kw={}):
    """
    Plot some data, with an option to plot a best fit model on top. Only the
    real part of the data (and model if provided) is plotted for clarity.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be plotted.
        
    data : array_like or dict
        The data to be plotted. If a dict of spherical-harmonic modes, use the 
        spherical_mode argument to specify which mode to plot.
        
    xlim : array_like, optional
        The x-axis limits. The default is [-50,100].
        
    best_fit : dict, optional
        A best fit result dictionary containing the model to plot on top of
        the data. The 'model_times' and 'model' entries are accessed. If the
        model is a dictionary of spherical-harmonic modes, use the 
        spherical_mode argument to specify which mode to plot. If None, no 
        bestfit model is plotted. The default is None.
        
    spherical_mode : tuple, optional
        A (l,m) tuple to specify which spherical harmonic mode to plot. The 
        default is None.
        
    log : bool, optional
        If True, use a log scale for the y-axis. This also plots the absolute
        value of the data. The default is False.
        
    outfile : str, optional
        File name/path to save the figure. If None, the figure is not saved. 
        The default is None.
        
    fig_kw : dict, optional
        Additional keyword arguments to pass to plt.subplots() at the figure
        creation. The default is {}.
    """
    if type(data) == dict:
        if spherical_mode is None:
            print("""Please specify the spherical mode to plot with the 
                  spherical_mode argument.""")
        else:
            data = data[spherical_mode]
            
    # We only plot the real part
    data = np.real(data)
            
    if log:
        data = abs(data)
        
    fig, ax = plt.subplots(figsize=(8,4), **fig_kw)
    
    ax.plot(times, data, 'k-', label='Re[data]')

    if best_fit is not None:
        
        if type(best_fit['model']) == dict:
        
            if spherical_mode is None:
                print("""Please specify the best fit spherical mode to plot 
                      with the spherical_mode argument.""")
            
            else:
                model_data = best_fit['model'][spherical_mode]
                  
        else:
            model_data = best_fit['model']
            
        model_data = np.real(model_data)
        
        if log:
            model_data = abs(model_data)
            
        ax.plot(
            best_fit['model_times'], model_data, 'r-', label='Re[model]', 
            alpha=0.8
            )

    ax.set_xlim(xlim[0],xlim[1])
    ax.set_xlabel('$t\ [M]$')
    if spherical_mode is None:
        ax.set_ylabel('$h$')
    else:
        ax.set_ylabel(f'$h_{{{spherical_mode[0]}{spherical_mode[1]}}}$')
        
    if log:
        ax.set_yscale('log')

    ax.legend(frameon=False)
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()
        
        
def plot_ringdown_modes(best_fit, spherical_mode=None, plot_type='re', 
                        xlim=None, ylim=None, legend=True, outfile=None, 
                        fig_kw={}):
    """
    Plot the ringdown waveform from a least-squares fit, decomposed into its
    individual modes.

    Parameters
    ----------
    bestfit : dict
        A best fit result dictionary containing the model to plot. If the
        model is a dictionary of spherical-harmonic modes, use the 
        spherical_mode argument to specify which mode to plot. 
    
    spherical_mode : tuple, optional
        A (l,m) tuple to specify which spherical harmonic mode to plot. The 
        default is None.
        
    plot_type : str, optional
        Specify whether to plot the real ('re') or imaginary ('im') part of
        the waveform. The default is 're'.
        
    xlim : array_like, optional
        The x-axis limits. The default is None.
        
    ylim : array_like, optional
        The y-axis limits. The default is None.
        
    legend : bool, optional
        Toggle the legend on or off. The default is True (legend on).
        
    outfile : str, optional
        File name/path to save the figure. If None, the figure is not saved. 
        The default is None.
        
    fig_kw : dict, optional
        Additional keyword arguments to pass to plt.subplots() at the figure
        creation. The default is {}.
    """
    fig, ax = plt.subplots(figsize=(8,4), **fig_kw)
    
    # Initialize an array to manually sum the modes on as a check, and get the 
    # relevant complex amplitudes
    if type(best_fit['model']) == dict:
        if spherical_mode is None:
            print("""Please specify the spherical mode to plot with the 
                  spherical_mode argument.""")
        else:
            mode_sum = np.zeros_like(best_fit['model'][spherical_mode])
            complex_amplitudes = best_fit['weighted_C'][spherical_mode]
    
    else:
        mode_sum = np.zeros_like(best_fit['model'])
        complex_amplitudes = best_fit['C']
    
    for i in range(len(best_fit['modes'])):
        
        # The waveform for each mode
        mode_waveform = ringdown(
            best_fit['model_times'], 
            best_fit['t0'], 
            [complex_amplitudes[i]], 
            [best_fit['frequencies'][i]]
            )
        
        # Add to the overall sum
        mode_sum += mode_waveform
        
        # Use a reduced opacity color if the color cycle repeats
        if i > 9:
            alpha = 0.5
        else:
            alpha = 0.7
        
        # Add the mode waveform to the figure
        if plot_type == 're':
            ax.plot(best_fit['model_times'], np.real(mode_waveform), alpha=alpha)
        elif plot_type == 'im':
            ax.plot(best_fit['model_times'], np.imag(mode_waveform), alpha=alpha)
    
    # The overall sum
    if plot_type == 're':
        ax.plot(best_fit['model_times'], np.real(mode_sum), 'k--')
    elif plot_type == 'im':
        ax.plot(best_fit['model_times'], np.imag(mode_sum), 'k--')
    
    if xlim is not None:
        ax.set_xlim(xlim[0],xlim[1])
    ax.set_xlabel('$t\ [M]$')
    
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
        
    if spherical_mode is None:
        ax.set_ylabel('$h$')
    else:
        ax.set_ylabel(f'$h_{{{spherical_mode[0]}{spherical_mode[1]}}}$')
    
    # Generate the list of labels for the legend
    labels = best_fit['mode_labels'].copy()
    labels.append('Sum')
    
    if legend:
        ax.legend(ax.lines, labels, ncol=3)
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()
        
        
def plot_mode_amplitudes(coefficients, labels, log=False, outfile=None, 
                         fig_kw={}):
    """
    Plot the magnitudes of the ringdown modes from a least-squares fit.

    Parameters
    ----------
    coefficients : array_like
        The complex coefficients from a ringdown fit. These are stored as
        best_fit['C'] and best_fit['weighted_C'].
        
    labels : array_like
        The labels for each coefficient. These are stored as 
        best_fit['mode_labels'].
    
    log : bool, optional
        If True, use a log scale for the amplitudes. The default is False.
        
    outfile : str, optional
        File name/path to save the figure. If None, the figure is not saved. 
        The default is None.
        
    fig_kw : dict, optional
        Additional keyword arguments to pass to plt.subplots() at the figure
        creation. The default is {}.
    """
    # Get the amplitudes from the complex coefficients
    amplitudes = abs(coefficients)
    
    # x-axis values, useful for plotting
    x = np.arange(len(amplitudes))
        
    # Create figure
    if len(x) > 24:
        figsize = (len(x)*0.3, 4)
    else:
        figsize = (6,4)
        
    fig, ax = plt.subplots(figsize=figsize, **fig_kw)
    
    for i in range(len(amplitudes)):
        ax.plot(
            [x[i],x[i]], [0,amplitudes[i]], color=f'C{i}', marker='o', 
            markevery=(1,2), linestyle=':')
    
    if log:
        ax.set_yscale('log')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=90)
    
    ax.set_xlabel('Mode')
    ax.set_ylabel('$|C|$')
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()


def mismatch_t0_array(times, data, modes, Mf, chif, t0_array, t0_method='geq', 
                      T_array=100, spherical_modes=None, delta=0):
    """
    Calculate the mismatch for an array of start times.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data_dict : array_like or dict
        The data to be fitted by the ringdown model. If a dict of 
        spherical-harmonic modes, use the spherical_modes argument to specify
        which modes to include in the fit.
        
    modes : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to include in 
        the ringdown model. For regular (positive real part) modes use 
        sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
        
    Mf : float or array_like
        The remnant black hole mass, which along with chif determines the QNM
        frequencies. This can be a float, so that mass doesn't change with
        time, or an array of the same length as times.
        
    chif : float or array_like
        The magnitude of the remnant black hole spin. As with Mf, this can be
        a float or an array.
        
    t0_array : array_like
        The start times of the ringdown model.
        
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
        
    T_array : float or array, optional
        The duration of the data to analyse, such that the end time is t0 + T. 
        If an array, this should be the same length as t0_array. The default 
        is 100.
        
    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical-harmonic modes 
        the analysis should be performed on. If None, all the modes contained 
        in data_dict are used. The default is None.

    delta : float or array_like, optional
        Modify the frequencies used in the ringdown fit. Either a
        constant value to modify every overtone frequency identically, or 
        an array with different values for each overtone.  Default is 0 
        (no modification). Only used if not using multimode_ringdown_fit.

    Returns
    -------
    mm_list : ndarray
        The mismatch for each t0 value.
    """
    # List to store the mismatch from each choice of t0
    mm_list = []
    
    # Make T array-like if a float is provided
    if type(T_array) != np.ndarray:
        T_array = T_array*np.ones(len(t0_array))
    
    # Fits with a fixed Kerr spectrum
    # -------------------------------
    
    if (type(Mf) in [float, np.float64]) & (type(chif) in [float, np.float64]):
        
        if type(data) == dict:
            for t0, T in zip(t0_array, T_array):
                best_fit = multimode_ringdown_fit(
                    times, data, modes, Mf, chif, t0, t0_method, T, 
                    spherical_modes)
                mm_list.append(best_fit['mismatch'])
                
        else:
            for t0, T in zip(t0_array, T_array):
                best_fit = ringdown_fit(
                    times, data, modes, Mf, chif, t0, t0_method, T,delta)
                mm_list.append(best_fit['mismatch'])
                
    # Fits with a dynamic Kerr spectrum
    # ---------------------------------
    
    else:
        
        if type(data) == dict:
            for t0, T in zip(t0_array, T_array):
                best_fit = dynamic_multimode_ringdown_fit(
                    times, data, modes, Mf, chif, t0, t0_method, T, 
                    spherical_modes)
                mm_list.append(best_fit['mismatch'])
                
        else:
            for t0, T in zip(t0_array, T_array):
                best_fit = dynamic_ringdown_fit(
                    times, data, modes, Mf, chif, t0, t0_method, T)
                mm_list.append(best_fit['mismatch'])
        
    return mm_list


def mismatch_M_chi_grid(times, data, modes, Mf_minmax, chif_minmax, t0, 
                        t0_method='geq', T=100, res=50, spherical_modes=None, delta =0):
    """
    Calculate the mismatch for a grid of Mf and chif values.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data_dict : array_like or dict
        The data to be fitted by the ringdown model. If a dict of 
        spherical-harmonic modes, use the spherical_modes argument to specify
        which modes to include in the fit.
        
    modes : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to include in 
        the ringdown model. For regular (positive real part) modes use 
        sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
    
    Mf_minmax : tuple
        The minimum and maximum values for the mass to use in the grid.
        
    chif_minmax : tuple
        The minimum and maximum values for the dimensionless spin magnitude to
        use in the grid.
        
    t0 : float
        The start time of the ringdown model.
        
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
        
    res : int, optional
        The number of points used along each axis of the grid (so there are
        res^2 evaluations of the mismatch). The default is 50.
        
    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical-harmonic modes 
        the analysis should be performed on. If None, all the modes contained 
        in data_dict are used. The default is None.

    delta : float or array_like (optional)
        Modify the frequencies used in the ringdown fit. Either a
        constant value to modify every overtone frequency identically, or 
        an array with different values for each overtone.  Default is 0 
        (no modification). Only used if using ringdown_fit for data.
        
    Returns
    -------
    mm_grid : ndarray
        The mismatch for each mass-spin combination.
    """
    # Create the mass and spin arrays
    Mf_array = np.linspace(Mf_minmax[0], Mf_minmax[1], res)
    chif_array = np.linspace(chif_minmax[0], chif_minmax[1], res)

    # List to store the mismatch from each choice of M and chi
    mm_list = []
    
    # Cycle through each combination of mass and spin, calculating the
    # mismatch for each. Use a single loop for the progress bar.
    if type(data) == dict:
        for i in tqdm(range(len(Mf_array)*len(chif_array))):
    
            Mf = Mf_array[int(i/len(Mf_array))]
            chif = chif_array[i%len(chif_array)]
        
            best_fit = multimode_ringdown_fit(
                times, data, modes, Mf, chif, t0, t0_method, T, 
                spherical_modes)
            mm_list.append(best_fit['mismatch'])
            
    else:
        for i in tqdm(range(len(Mf_array)*len(chif_array))):
    
            Mf = Mf_array[int(i/len(Mf_array))]
            chif = chif_array[i%len(chif_array)]
        
            best_fit = ringdown_fit(
                times, data, modes, Mf, chif, t0, t0_method, T ,delta)
            mm_list.append(best_fit['mismatch'])

    # Convert the list of mismatches to a grid
    mm_grid = np.reshape(np.array(mm_list), (len(Mf_array), len(chif_array)))
    
    return mm_grid


def calculate_epsilon(times, data, modes, Mf, chif, t0, t0_method='geq', T=100, 
                      spherical_modes=None, min_method='Nelder-Mead',delta=0):
    r"""
    Find the Mf and chif values that minimize the mismatch for a given
    ringdown start time and model, and from this calculate the 'distance' of 
    the best fit mass and spin values from the true remnant properties 
    (expressed through epsilon).

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data_dict : array_like or dict
        The data to be fitted by the ringdown model. If a dict of 
        spherical-harmonic modes, use the spherical_modes argument to specify
        which modes to include in the fit.
        
    modes : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to include in 
        the ringdown model. For regular (positive real part) modes use 
        sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
        
    Mf : float
        The remnant black hole mass. Along with calculating epsilon, this is
        used for the initial guess in the minimization.
        
    chif : float
        The magnitude of the remnant black hole spin. Along with calculating 
        epsilon, this is used for the initial guess in the minimization.
        
    t0 : float
        The start time of the ringdown model.
        
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
        
    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical-harmonic modes 
        the analysis should be performed on. If None, all the modes contained 
        in data_dict are used. The default is None.
        
    min_method : str, optional
        The method used to find the mismatch minimum in the mass-spin space.
        This can be any method available to scipy.optimize.minimize. This 
        includes None, in which case the method is automatically chosen. The
        default is 'Nelder-Mead'.

    delta : float or array_like (optional)
        Modify the frequencies used in the ringdown fit. Either a
        constant value to modify every overtone frequency identically, or 
        an array with different values for each overtone.  Default is 0 
        (no modification). Only used if using ringdown_fit.

    Returns
    -------
    epsilon : float
        The difference between the true Mf and chif values and values that 
        minimize the mismatch. Defined as 
        
        .. math::
            \epsilon = \sqrt{ \left( \delta M_f \right)^2 + 
                              \left( \delta\chi_f \right)^2 }.
            
        where :math:`\delta M_f = M_\mathrm{best fit} - M_f` and 
        :math:`\delta \chi_f = \chi_\mathrm{best fit} - \chi_f`
        
    Mf_bestfit: float
        The remnant mass that minimizes the mismatch.
             
    chif_bestfit : float
        The remnant spin that minimizes the mismatch.
    """ 
    # The initial guess in the minimization
    x0 = [Mf, chif]
    
    # Other settings for the minimzation
    bounds = [(0,1.5), (0,0.99)]
    options = {'xatol':1e-6,'disp':False}
    
    if type(data) == dict:
        
        def mismatch_M_chi(x, times, data_dict, modes, t0, t0_method, T, 
                           spherical_modes):
            """
            A wrapper for the multimode_ringdown_fit function, for use with 
            the SciPy minimize function.
            """
            Mf = x[0]
            chif = x[1]
            
            if chif > 0.99:
                chif = 0.99
            if chif < 0:
                chif = 0
            
            best_fit = multimode_ringdown_fit(
                times, data_dict, modes, Mf, chif, t0, t0_method, T, 
                spherical_modes)
            
            return best_fit['mismatch']
        
        # Perform the SciPy minimization
        res = minimize(
            mismatch_M_chi, 
            x0,
            args=(times, data, modes, t0, t0_method, T, spherical_modes),
            method=min_method, 
            bounds=bounds, 
            options=options
            )
        
    else:
        
        def mismatch_M_chi(x, times, data_dict, modes, t0, t0_method, T,delta):
            """
            A wrapper for the ringdown_fit function, for use with the SciPy 
            minimize function.
            """
            Mf = x[0]
            chif = x[1]
            
            if chif > 0.99:
                chif = 0.99
            if chif < 0:
                chif = 0
            
            best_fit = ringdown_fit(
                times, data_dict, modes, Mf, chif, t0, t0_method, T,delta)
            
            return best_fit['mismatch']
        
        # Perform the SciPy minimization
        res = minimize(
            mismatch_M_chi, 
            x0,
            args=(times, data, modes, t0, t0_method, T),
            method=min_method, 
            bounds=bounds, 
            options=options
            )

    # The remnant properties that give the minimum mismatch
    Mf_bestfit = res.x[0]
    chif_bestfit = res.x[1]
        
    # Calculate epsilon
    delta_Mf = Mf_bestfit - Mf
    delta_chif = chif_bestfit - chif
    epsilon = np.sqrt(delta_Mf**2 + delta_chif**2)
        
    return epsilon, Mf_bestfit, chif_bestfit


def plot_mismatch_M_chi_grid(mm_grid, Mf_minmax, chif_minmax, truth=None, 
                             marker=None, outfile=None, fig_kw={}):
    """
    Helper function to plot the mismatch grid (from a call of the 
    mismatch_M_chi_grid) as a heatmap with a colorbar. There are also options 
    to indicate the true mass and spins, and to highlight a particular 
    mass-spin combination.

    Parameters
    ----------
    mm_grid : array_like
        A 2D array of mismatches in mass-spin space.
        
    Mf_minmax : tuple
        The minimum and maximum values of the mass used in the grid.
        
    chif_minmax : tuple
        The minimum and maximum values of the dimensionless spin magnitude
        used in the grid.
        
    truth : tuple, optional
        A tuple of the form (Mf_true, chif_mag_true) to indicate the true
        remnant properties on the heatmap. If None, nothing is plotted. The
        default is None.
        
    marker : tuple, optional
        A tuple of the form (Mf, chif) to indicate a particular mass-spin
        combination. For example, this could be used to indicate the best fit
        mass and spin from a call of calculate_epsilon. If None, nothing is 
        plotted. The default is None.
    
    outfile : str, optional
        File name/path to save the figure. If None, the figure is not saved. 
        The default is None.
        
    fig_kw : dict, optional
        Additional keyword arguments to pass to plt.subplots() at the figure
        creation. The default is {}.
    """
    Mf_min, Mf_max = Mf_minmax
    chif_min, chif_max = chif_minmax
    
    fig, ax = plt.subplots(**fig_kw)
    
    # Plot heatmap
    im = ax.imshow(
        np.log10(mm_grid), 
        extent=[chif_min,chif_max,Mf_min,Mf_max],
        aspect='auto',
        origin='lower',
        interpolation='bicubic',
        cmap='gist_heat_r')

    if truth is not None:
        # Indicate true values
        ax.axhline(truth[0], color='w', alpha=0.3)
        ax.axvline(truth[1], color='w', alpha=0.3)
        
    if marker is not None:
        # Mark a particular mass-spin combination
        ax.plot(marker[0], marker[1], marker='o', markersize=3, color='k')

    # Color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('$\mathrm{log}_{10}\ \mathcal{M}$')

    ax.set_xlabel('$\chi_f$')
    ax.set_ylabel('$M_f\ [M]$')
    
    plt.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()


def mismatch_omega_grid(times, data, modes, Mf, chif, re_minmax, im_minmax, t0, 
                        t0_method='geq', T=100, res=50):
    """
    Calculate the mismatch for a grid of complex frequencies.

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
    
    re_minmax : tuple
        The minimum and maximum values for the real part of the complex 
        frequency to use in the grid.
    
    im_minmax : tuple
        The minimum and maximum values for the imaginary part of the complex 
        frequency to use in the grid.

    t0 : float
        The start time of the ringdown model.

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

    res : int, optional
        The number of points used along each axis of the grid (so there are
        res^2 evaluations of the mismatch). The default is 50.

    Returns
    -------
    mm_grid : ndarray
        The mismatch for each complex frequency.
    """
    # Create the real and imaginary omega arrays
    re_array = np.linspace(re_minmax[0], re_minmax[1], res)
    im_array = np.linspace(im_minmax[0], im_minmax[1], res)

    # List to store the mismatch from each complex frequency
    mm_list = []
    
    for i in tqdm(range(len(re_array)*len(im_array))):

        re = re_array[int(i/len(re_array))]
        im = im_array[i%len(im_array)]

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
            print("""Requested t0_method is not valid. Please choose between 
                  'geq' and 'closest'""")
        
        # Frequencies
        # -----------
        
        frequencies = np.array(qnm.omega_list(modes, chif, Mf) + [re+1j*im])
            
        # Construct coefficient matrix and solve
        # --------------------------------------
        
        # Construct the coefficient matrix
        a = np.array([
            np.exp(-1j*frequencies[i]*(times-t0)) for i in range(len(frequencies))
            ]).T

        # Solve for the complex amplitudes, C. Also returns the sum of 
        # residuals, the rank of a, and singular values of a.
        C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
        
        # Evaluate the model
        model = np.einsum('ij,j->i', a, C)
        
        # Calculate the mismatch for the fit
        mm = mismatch(times, model, data)
        
        # Create a list of mode labels (can be used for plotting)
        labels = [str(mode) for mode in modes]
        
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
            'frequencies': frequencies
            }
    
        mm_list.append(best_fit['mismatch'])

    # Convert the list of mismatches to a grid
    mm_grid = np.reshape(np.array(mm_list), (len(re_array), len(im_array))).T
    
    return mm_grid


def plot_mismatch_omega_grid(mm_grid, re_minmax, im_minmax, truth=None, 
                             marker=None, outfile=None, fig_kw={}):
    """
    Helper function to plot the mismatch grid (from a call of the
    mismatch_omega_grid) as a heatmap with a colorbar.

    Parameters
    ----------
    mm_grid : array_like
        A 2D array of mismatches in complex-frequency space.

    re_minmax : tuple
        The minimum and maximum values of the real part of the complex 
        frequency used in the grid.

    im_minmax : tuple
        The minimum and maximum values of the imaginary part of the complex 
        frequency used in the grid.

    truth : tuple, optional
        A tuple of the form (re_true, im_true) to indicate the true
        complex frequency on the heatmap. If None, nothing is plotted. The
        default is None.

    marker : tuple, optional
        A tuple of the form (re, im) to indicate a particular complex 
        frequency. If None, nothing is plotted. The default is None.

    outfile : str, optional
        File name/path to save the figure. If None, the figure is not saved. 
        The default is None.

    fig_kw : dict, optional
        Additional keyword arguments to pass to plt.subplots() at the figure
        creation. The default is {}.
    """
    re_min, re_max = re_minmax
    im_min, im_max = im_minmax
    
    fig, ax = plt.subplots(**fig_kw)
    
    # Plot heatmap
    im = ax.imshow(
        np.log10(mm_grid), 
        extent=[re_min,re_max,im_min,im_max],
        aspect='auto',
        origin='lower',
        interpolation='bicubic',
        cmap='gist_heat_r')

    if truth is not None:
        # Indicate true values
        ax.axhline(truth[0], color='w', alpha=0.3)
        ax.axvline(truth[1], color='w', alpha=0.3)
        
    if marker is not None:
        # Mark a partiular mass-spin combination
        ax.plot(marker[0], marker[1], marker='o', markersize=3, color='k')

    # Color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('$\mathrm{log}_{10}\ \mathcal{M}$')

    ax.set_xlabel('$\mathrm{Re}[\omega]$')
    ax.set_ylabel('$\mathrm{Im}[\omega]$')
    
    plt.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()
        
        
def free_frequency_fit(times, data, t0, modes=[], Mf=None, chif=None, 
                       t0_method='geq', T=100, min_method='Nelder-Mead'):
    """
    Find the complex frequency that minimizes the mismatch for a given 
    ringdown start time. A set of "fixed" frequencies can also be included in 
    the fit, specified through the modes, Mf, and chif arguments.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data : array_like
        The data to be fitted by the ringdown model.
        
    t0 : float
        The start time of the ringdown model.
        
    modes : array_like, optional
        A sequence of (l,m,n,sign) tuples to specify which (fixed) QNMs to 
        include in the ringdown model. For regular (positive real part) modes 
        use sign=+1. For mirror (negative real part) modes use sign=-1. For 
        nonlinear modes, the tuple has the form 
        (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
        
    Mf : float, optional
        The remnant black hole mass, which along with chif determines the QNM
        frequencies.
        
    chif : float, optional
        The magnitude of the remnant black hole spin.
        
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
        
    min_method : str, optional
        The method used to find the mismatch minimum in the complex-frequency 
        space. This can be any method available to scipy.optimize.minimize. 
        This includes None, in which case the method is automatically chosen. 
        The default is 'Nelder-Mead'.

    Returns
    -------
    omega_bestfit : complex
        The complex frequency that minimizes the mismatch.
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
              
    # Compute fixed frequencies - we always include these in the fit (along 
    # with the additional free frequency)
    fixed_frequencies = np.array(qnm.omega_list(modes, chif, Mf))
    
    # The initial guess in the minimization
    x0 = [1, -0.5]
    
    # Other settings for the minimzation
    bounds = [(0,2), (-1,0)]
    options = {'xatol':1e-8,'disp':False}
    
    def mismatch_f_tau(x, times, data, t0):
        """
        A modification to the ringdown_fit function which allows a single 
        frequency to be varied, for use with the SciPy minimize function.
        """
        # Extract the value of the free frequency
        omega_free = x[0] + 1j*x[1]
        
        # Combine the free frequency with the fixed frequencies
        frequencies = np.hstack([fixed_frequencies, omega_free])
        
        # Construct the coefficient matrix
        a = np.array([
            np.exp(-1j*frequencies[i]*(times-t0)) for i in range(len(frequencies))
            ]).T
    
        # Solve for the complex amplitudes, C. Also returns the sum of 
        # residuals, the rank of a, and singular values of a.
        C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
    
        # Evaluate the model
        model = np.einsum('ij,j->i', a, C)
        
        # Calculate the mismatch for the fit
        mm = mismatch(times, model, data)
        
        return mm
    
    # Perform the SciPy minimization
    res = minimize(
        mismatch_f_tau, 
        x0,
        args=(times, data, t0),
        method=min_method, 
        bounds=bounds, 
        options=options
        )
    
    omega_bestfit = res.x[0] + 1j*res.x[1]
    
    return omega_bestfit
    

def rational_filter(times, data, modes, Mf, chif, t_start=-300, t_end=None, 
                    dt=None, t_taper=100, align_inspiral=True):
    """
    This function applies the rational filter described by [#] to remove the
    specified QNM content from some data. The data is then time-shifted so 
    that the early (inspiral) part of the data is left unaffected.
    
    Because Fourier transforms are involved, it is first necessary to
    interpolate the data onto a regularly spaced array of times and to apply a
    tapering window at early times. This is what the t_start, t_end, dt and 
    t_taper arguments refer to.
    
    [#] S. Ma, K. Mitman, L Sun et al (2022) arXiv:2207.10870 [gr-qc]
     
    Parameters
    ----------
    times : array_like
        The times associated with the data to be filtered.
        
    data : array_like
        The data to be filtered.
        
    modes : array_like
        A sequence of (l,m,n,sign) tuples to specify which QNMs to filter. For 
        regular (positive real part) modes use sign=+1. For mirror (negative 
        real part) modes use sign=-1.
        
    Mf : float
        The remnant black hole mass, which along with chif determines the QNM
        frequencies.
        
    chif : float
        The magnitude of the remnant black hole spin.
        
    t_start, t_end, dt, t_taper : float
        Use regularly sampled times in range t_start to t_end (if None,
        then defaults to end of signal) with time step dt (if None, then 
        defaults to the minimum time step in times). The start of the signal 
        is tapered smoothly to zero, and the length of signal effected is 
        t_taper. The defaults are -300, None, None.
        
    align_inspiral: bool
        Controls if the time shift to align the inspiral is applied. The 
        default is True.
        
    Returns
    -------
    uniform_times : ndarray
        Array of regularly spaced sampling times at which the filtered data
        is evaluated.
        
    filtered_data : ndarray
        The data filtered to remove the desired QNM content and time shifted
        such that it agrees with the original data at early times (as 
        described in Ref [#]).
    """
    # Default to the end of the data
    if t_end is None:
        t_end = times[-1]
        
    # Default to the minimum time spacing
    if dt is None:
        dt = min(np.diff(times))
        
    # Interpolate data onto regular grid of times
    uniform_times = np.arange(t_start, t_end, dt)
    uniform_data = interp1d(times, data.real, kind='cubic')(uniform_times) \
        + 1j*interp1d(times, data.imag, kind='cubic')(uniform_times)

    # Smoothly taper interpolated signal to zero at early times to avoid
    # possible problems with the Fourier transform
    
    # Mask to isolate the data we will apply the taper to
    taper_mask = uniform_times < (t_start+t_taper)
    
    # Construct taper
    taper_length = np.sum(taper_mask)
    taper_arg = np.pi*np.arange(taper_length)[::-1]/taper_length
    taper = (np.cos(taper_arg) + 1)/2
    
    # Apply to the data
    uniform_data[taper_mask] *= taper

    # Forward Fourier transform
    freqs = np.fft.fftfreq(len(uniform_data), d=dt)
    fourier_data = np.fft.fft(uniform_data)

    # Construct the rational filter
    filt = np.ones_like(fourier_data)
    phase_shift, time_shift = 0., 0.
    for l, m, n, sign in modes:
        omega = qnm.omega(l, m, n, sign, chif, Mf)
        filt *= (2*np.pi*freqs+omega)/(2*np.pi*freqs+np.conj(omega))
        phase_shift += np.angle(omega/np.conj(omega))
        time_shift += np.abs(2*np.imag(omega)/np.conj(omega)**2)

    # Apply the filter
    fourier_data *= filt

    # Apply time shift to realign the inspiral
    if align_inspiral:
        fourier_data *= np.exp(-2*np.pi*(1j)*freqs*time_shift-(1j)*phase_shift)

    # Inverse Fourier transform
    filtered_data = np.fft.ifft(fourier_data)

    return uniform_times, filtered_data
