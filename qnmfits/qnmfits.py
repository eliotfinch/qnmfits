import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

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


def ringdown_fit(times, data, modes, Mf, chif, t0, mirror_modes=[], 
                 t0_method='geq', T=100):
    """
    Perform a least-squares fit to some data using a ringdown model.

    Parameters
    ----------
    times : array_like
        The times associated with the data to be fitted.
        
    data : array_like
        The data to be fitted by the ringdown model.
        
    modes : array_like
        A sequence of (l,m,n) tuples to specify which regular (positive real 
        part) QNMs to include in the ringdown model. For nonlinear modes, the 
        tuple has the form (l1,m1,n1,l2,m2,n2,...).
        
    Mf : float
        The remnant black hole mass, which along with chif determines the QNM
        frequencies.
        
    chif : float
        The magnitude of the remnant black hole spin.
        
    t0 : float
        The start time of the ringdown model.
        
    mirror_modes : array_like, optional
        The same as modes, but for the mirror (negative real part) QNMs. The 
        default is [] (no mirror modes are included).
        
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
        The end time of the analysis, relative to t0. The default is 100.

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
                The regular ringdown modes used in the fit.
            - 'mirror_modes' : ndarray
                The mirror ringdown modes used in the fit.
            - 'mode_labels' : list
                Labels for each of the ringdown modes (used for plotting).
            - 'frequencies' : ndarray
                The values of the complex frequencies for all the ringdown 
                modes. The order is [modes, mirror_modes].
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
    
    # The regular (positive real part) frequencies
    reg_frequencies = np.array(qnm.omega_list(modes, chif, Mf, interp=True))
    
    # The mirror (negative real part) frequencies can be obtained using 
    # symmetry properties 
    mirror_frequencies = -np.conjugate(qnm.omega_list(
        [(l,-m,n) for l,m,n in mirror_modes], chif, Mf))
    
    frequencies = np.hstack((reg_frequencies, mirror_frequencies))
        
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
    labels = []
    for mode in modes:
        labels.append(str(mode))
    for mode in mirror_modes:
        labels.append(str(mode) + '$^\prime$')
    
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
        'mirror_modes': mirror_modes,
        'mode_labels': labels,
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit

    
def multimode_ringdown_fit(times, data_dict, modes, Mf, chif, t0, 
                           mirror_modes=[], t0_method='geq', T=100, 
                           spherical_modes=None):
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
        A sequence of (l,m,n) tuples to specify which regular (positive real 
        part) QNMs to include in the ringdown model. For nonlinear modes, the 
        tuple has the form (l1,m1,n1,l2,m2,n2,...).
        
    Mf : float
        The remnant black hole mass, which along with chif determines the QNM
        frequencies.
        
    chif : float
        The magnitude of the remnant black hole spin.
        
    t0 : float
        The start time of the ringdown model.
        
    mirror_modes : array_like, optional
        The same as modes, but for the mirror (negative real part) QNMs. The 
        default is [] (no mirror modes are included).
        
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
        The end time of the analysis, relative to t0. The default is 100.
        
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
                The regular ringdown modes used in the fit.
            - 'mirror_modes' : ndarray
                The mirror ringdown modes used in the fit.
            - 'mode_labels' : list
                Labels for each of the ringdown modes (used for plotting).
            - 'frequencies' : ndarray
                The values of the complex frequencies for all the ringdown 
                modes. The order is [modes, mirror_modes].
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
    
    # The regular (positive real part) frequencies
    reg_frequencies = np.array(qnm.omega_list(modes, chif, Mf, interp=True))
    
    # The mirror (negative real part) frequencies can be obtained using 
    # symmetry properties 
    mirror_frequencies = -np.conjugate(qnm.omega_list(
        [(l,-m,n) for l,m,n in mirror_modes], chif, Mf))
    
    frequencies = np.hstack((reg_frequencies, mirror_frequencies))
    
    # Construct the coefficient matrix for use with NumPy's lstsq 
    # function. We deal with the regular mode and mirror mode mixing
    # coefficients separately.
    
    # Regular mixing coefficients
    # ---------------------------
    
    # A list of lists for the mixing coefficient indices. The first 
    # list is associated with the first hlm mode. The second list is 
    # associated with the second hlm mode, and so on.
    # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
    #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
    reg_indices_lists = [
        [lm_mode+mode for mode in modes] for lm_mode in spherical_modes]
    
    # Convert each tuple of indices in indices_lists to a mu value
    reg_mu_lists = np.conjugate([
        qnm.mu_list(indices, chif, interp=True) for indices in reg_indices_lists])
    
    # Mirror mixing coefficients
    # --------------------------
        
    # A list of lists for the mixing coefficient indices, see above
    mirror_indices_lists = [
        [(l,-m)+(L,-M,N) for L,M,N in mirror_modes] for l,m in spherical_modes]
    
    # We need to multiply each mu by a factor (-1)**(l+l'). Construct these
    # factors from the indices_lists.
    signs = [np.array([
        (-1)**(indices[0]+indices[2]) for indices in indices_list]) 
        for indices_list in mirror_indices_lists]
    
    # Convert each tuple of indices in indices_lists to a mu value
    mirror_mu_lists = [
        signs[i]*np.array(qnm.mu_list(indices, chif)) 
        for i, indices in enumerate(mirror_indices_lists)]
    
    # Combine the regular and mirror mixing coefficients
    mu_lists = [
        list(reg_mu_lists[i]) + list(mirror_mu_lists[i]) 
        for i in range(len(spherical_modes))]
        
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
    labels = []
    for mode in modes:
        labels.append(str(mode))
    for mode in mirror_modes:
        labels.append(str(mode) + '$^\prime$')
    
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
        'mirror_modes': mirror_modes,
        'mode_labels': labels,
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit


def dynamic_multimode_ringdown_fit(times, data_dict, modes, Mf, chif, t0, 
                                   mirror_modes=[], t0_method='geq', T=100, 
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
        A sequence of (l,m,n) tuples to specify which regular (positive real 
        part) QNMs to include in the ringdown model. For nonlinear modes, the 
        tuple has the form (l1,m1,n1,l2,m2,n2,...).
        
    Mf : float or array_like
        The remnant black hole mass, which along with chif determines the QNM
        frequencies. This can be a float, so that mass doesn't change with
        time, or an array of the same length as times.
        
    chif : float or array_like
        The magnitude of the remnant black hole spin. As with Mf, this can be
        a float or an array.
        
    t0 : float
        The start time of the ringdown model.
        
    mirror_modes : array_like, optional
        The same as modes, but for the mirror (negative real part) QNMs. The 
        default is [] (no mirror modes are included).
        
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
        The end time of the analysis, relative to t0. The default is 100.
        
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
                The regular ringdown modes used in the fit.
            - 'mirror_modes' : ndarray
                The mirror ringdown modes used in the fit.
            - 'mode_labels' : list
                Labels for each of the ringdown modes (used for plotting).
            - 'frequencies' : ndarray
                The values of the complex frequencies for all the ringdown 
                modes. The order is [modes, mirror_modes].
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

    Mf = Mf[data_mask]
    if type(chif) in [float, np.float64]:
        chif = np.full(len(times), chif)
    else:
        chif = chif[data_mask]
    
    # Frequencies
    # -----------
    
    # The regular (positive real part) frequencies
    reg_frequencies = np.array(qnm.omegaoft_list(modes, chif, Mf))
    
    # The mirror (negative real part) frequencies can be obtained using 
    # symmetry properties 
    mirror_frequencies = -np.conjugate(qnm.omegaoft_list(
        [(l,-m,n) for l,m,n in mirror_modes], chif, Mf))
    
    if len(mirror_modes) == 0:
        frequencies = reg_frequencies.T
    elif len(modes) == 0:
        frequencies = mirror_frequencies.T
    else:
        frequencies = np.hstack((reg_frequencies.T, mirror_frequencies.T))
    
    # We stack as many frequency arrays on top of each other as we have
    # spherical_modes
    frequencies = np.vstack(len(spherical_modes)*[frequencies])
        
    # Construct the coefficient matrix for use with NumPy's lstsq function. We 
    # deal with the regular mode and mirror mode mixing coefficients separately.
    
    # Regular mixing coefficients
    # ---------------------------
    
    if len(modes) != 0:
    
        # A list of lists for the mixing coefficient indices. The first 
        # list is associated with the first hlm mode. The second list is 
        # associated with the second hlm mode, and so on.
        # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
        #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
        reg_indices_lists = [
            [lm_mode+mode for mode in modes] for lm_mode in spherical_modes]
        
        # Convert each tuple of indices in indices_lists to an array of mu 
        # values
        reg_mu_lists = np.conjugate([
            qnm.muoft_list(indices, chif) for indices in reg_indices_lists])
        
        # I = len(spherical_modes)
        # J = len(modes) + len(mirror_modes)
        # K = len(times)
        
        # At this point, reg_mu_lists has a shape (I, J, K). We want to
        # reshape it into a 2D array of shape (I*K, J), such that the 
        # first K rows correspond to the first hlm mode.
        
        # Flatten to make reshaping easier
        reg_mu_lists = np.array([
            item for sublist in reg_mu_lists for item in sublist]).T
        
        # The above flattens the array into a 2D array of shape (K, I*J). 
        # So,the separate hlm mode arrays are stacked horizontally, and 
        # not in the desired vertical way.
        
        # Reshape
        reg_mu_lists = np.vstack([
            reg_mu_lists[:,i*len(modes):(i+1)*len(modes)] 
            for i in range(len(spherical_modes))])
        
        # The above reshaping converts the shape into the desired (I*K, J)
        
    if len(mirror_modes) != 0:
    
        # Mirror mixing coefficients
        # --------------------------
            
        # A list of lists for the mixing coefficient indices, see above
        mirror_indices_lists = [
            [(l,-m)+(L,-M,N) for L,M,N in mirror_modes] for l,m in spherical_modes]
        
        # We need to multiply each mu by a factor (-1)**(l+l'). Construct these
        # factors from the indices_lists.
        signs = [np.array([
            (-1)**(indices[0]+indices[2]) for indices in indices_list]) 
            for indices_list in mirror_indices_lists]
        
        # Convert each tuple of indices in indices_lists to a mu value
        mirror_mu_lists = np.array([
            signs[i][:,None]*np.array(qnm.muoft_list(indices, chif))
            for i, indices in enumerate(mirror_indices_lists)])
        
        # Flatten
        mirror_mu_lists = np.array([
            item for sublist in mirror_mu_lists for item in sublist]).T
        
        # Reshape
        mirror_mu_lists = np.vstack([
            mirror_mu_lists[:,i*len(mirror_modes):(i+1)*len(mirror_modes)] 
            for i in range(len(spherical_modes))])
    
    if len(mirror_modes) == 0:
        mu_lists = reg_mu_lists
    elif len(modes) == 0:
        mu_lists = mirror_mu_lists
    else:
        # Combine the regular and mirror mixing coefficients
        mu_lists = np.hstack((reg_mu_lists, mirror_mu_lists))
    
    # Construct the coefficient matrix
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
    labels = []
    for mode in modes:
        labels.append(str(mode))
    for mode in mirror_modes:
        labels.append(str(mode) + '$^\prime$')
    
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
        'mirror_modes': mirror_modes,
        'mode_labels': labels,
        'frequencies': frequencies
        }
    
    # Return the output dictionary
    return best_fit


def plot_ringdown(times, data, xlim=[-50,100], best_fit=None, 
                  spherical_mode=None, outfile=None, fig_kw={}):
    """
    Plot some data, with an option to plot a best fit model on top.

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
    
    fig, ax = plt.subplots(figsize=(8,4), **fig_kw)
    
    ax.plot(times, np.real(data), 'k-', label=r'$h_+$')
    # ax.plot(self.times, -np.imag(data), 'k--', label=r'$h_\times$')

    if best_fit is not None:
        
        if type(best_fit['model'] == dict):
        
            if spherical_mode is None:
                print("""Please specify the best fit spherical mode to plot 
                      with the spherical_mode argument.""")
            
            else:
                model_data = best_fit['model'][spherical_mode]
                  
        else:
            model_data = best_fit['model']
            
        ax.plot(
            best_fit['model_times'], np.real(model_data), 'r-', 
            label=r'$h_+$ model', alpha=0.8)
        # ax.plot(
        #     best_fit['model_times'], -np.imag(model_data), 'r--',
        #     label=r'$h_\times$ model', alpha=0.8)

    ax.set_xlim(xlim[0],xlim[1])
    ax.set_xlabel('$t\ [M]$')
    if spherical_mode is None:
        ax.set_ylabel('$h$')
    else:
        ax.set_ylabel(f'$h_{{{spherical_mode[0]}{spherical_mode[1]}}}$')

    ax.legend(loc='upper right', frameon=False)
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()
        
        
def plot_ringdown_modes(self, best_fit, spherical_mode=None, xlim=None, 
                        ylim=None, legend=True, outfile=None, fig_kw={}):
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
    # relevent complex amplitudes
    if type(best_fit['model'] == dict):
        if spherical_mode is None:
            print("""Please specify the spherical mode to plot with the 
                  spherical_mode argument.""")
        else:
            mode_sum = np.zeros_like(best_fit['model'][spherical_mode])
            complex_amplitudes = best_fit['weighted_C'][spherical_mode]
    
    else:
        mode_sum = np.zeros_like(best_fit['model'])
        complex_amplitudes = best_fit['C']
    
    for i in range(len(best_fit['modes'] + best_fit['mirror_modes'])):
        
        # The waveform for each mode
        mode_waveform = ringdown(
            best_fit['model_times'], 
            best_fit['t0'], 
            [complex_amplitudes[i]], 
            [best_fit['frequencies'][i]]
            )
        
        # Add to the overall sum
        mode_sum += mode_waveform
        
        # Use a reduced opacity colour if the colour cycle repeats
        if i > 9:
            alpha = 0.5
        else:
            alpha = 0.7
        
        # Add the mode waveform to the figure. We just plot the real part for
        # clarity.
        ax.plot(best_fit['model_times'], np.real(mode_waveform), alpha=alpha)
    
    # The overall sum
    ax.plot(best_fit['model_times'], np.real(mode_sum), 'k--')
    
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
        
        
def plot_mode_amplitudes(self, coefficients, labels, log=False, outfile=None, 
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
        ax.set_xscale('log')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='x', rotation=90)
    
    ax.set_xlabel('Mode')
    ax.set_ylabel('$|C|$')
    
    if outfile is not None:
        plt.savefig(outfile)
        plt.close()


def mismatch_t0_array(times, data, modes, Mf, chif, t0_array, mirror_modes=[],
                      t0_method='geq', T_array=100, spherical_modes=None):
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
        A sequence of (l,m,n) tuples to specify which regular (positive real 
        part) QNMs to include in the ringdown model. For nonlinear modes, the 
        tuple has the form (l1,m1,n1,l2,m2,n2,...).
        
    Mf : float or array_like
        The remnant black hole mass, which along with chif determines the QNM
        frequencies. This can be a float, so that mass doesn't change with
        time, or an array of the same length as times.
        
    chif : float or array_like
        The magnitude of the remnant black hole spin. As with Mf, this can be
        a float or an array.
        
    t0_array : array_like
        The start times of the ringdown model.
        
    mirror_modes : array_like, optional
        The same as modes, but for the mirror (negative real part) QNMs. The 
        default is [] (no mirror modes are included).
        
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
        The end time of the analysis, relative to t0. If an array, this 
        should be the same length as t0_array. The default is 100.
        
    spherical_modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical-harmonic modes 
        the analysis should be performed on. If None, all the modes contained 
        in data_dict are used. The default is None.

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
    
    if (type(Mf) == float) & (type(chif) == float):
        
        if type(data) == dict:
            for t0, T in zip(t0_array, T_array):
                best_fit = multimode_ringdown_fit(
                    times, data, modes, Mf, chif, t0, mirror_modes, t0_method, 
                    T, spherical_modes)
                mm_list.append(best_fit['mismatch'])
                
        else:
            for t0, T in zip(t0_array, T_array):
                best_fit = ringdown_fit(
                    times, data, modes, Mf, chif, t0, mirror_modes, t0_method, 
                    T)
                mm_list.append(best_fit['mismatch'])
                
    # Fits with a dynamic Kerr spectrum
    # ---------------------------------
    
    else:
        
        if type(data) == dict:
            for t0, T in zip(t0_array, T_array):
                best_fit = dynamic_multimode_ringdown_fit(
                    times, data, modes, Mf, chif, t0, mirror_modes, t0_method, 
                    T, spherical_modes)
                mm_list.append(best_fit['mismatch'])
                
        else:
            print("""Single-mode fits with a dynamic Kerr spectrum are not yet
                  implemented""")
        
    return mm_list


def mismatch_M_chi_grid(self, Mf_minmax, chif_minmax, res=50, t0=0, T=100, 
                        modes=[(2,2,n) for n in range(8)], mirror_modes=[],
                        hlm_modes=[(2,2)]):
    """
    Calculate the mismatch for a grid of Mf and chif values.

    Parameters
    ----------
    Mf_minmax : tuple
        The minimum and maximum values for the mass to use in the grid.
        
    chif_minmax : tuple
        The minimum and maximum values for the dimensionless spin 
        magnitude to use in the grid.
        
    res : int, optional
        The number of points used along each axis of the grid (so there 
        are res^2 evaluations of the mismatch). The default is 50.
        
    t0 : float, optional
        The model start time to compute the mismatch for. The default is 0.
        
    T : float, optional
        The end time of the analysis, relative to t0. The default is 100.
        
    modes : list, optional
        A list of (l,m,n) tuples (where l is the angular number of the 
        mode, m is the azimuthal number, and n is the overtone number) 
        specifying which modes to use in the model. The default is 
        [(2,2,n) for n in range(8)].
        
    mirror_modes : list, optional
        A list of (l,m,n) tuples (as in modes), specifying which 'mirror
        modes' to use in the model. The default is [] (no mirror modes are
        included).
        
    hlm_modes : list, optional
        A list of (l,m) tuples to specify which spherical harmonic hlm 
        modes the analysis should be performed on. The default is [(2,2)].
        
    Returns
    -------
    mm_grid : array
        The mismatch for each mass-spin combination.
    """
    # Create the mass and spin arrays
    Mf_array = np.linspace(Mf_minmax[0], Mf_minmax[1], res)
    chif_array = np.linspace(chif_minmax[0], chif_minmax[1], res)

    # List to store the mismatch from each choice of M and chi
    mm_list = []

    # Cycle through each combination of mass and spin, calculating the
    # mismatch for each. Use a single loop for the progress bar.
    for i in tqdm(range(len(Mf_array)*len(chif_array))):

        # Get the mass and spin values from the value of i
        Mf = Mf_array[int(i/len(Mf_array))]
        chif = chif_array[i%len(chif_array)]
        
        # Run the fit
        bestfit = self.ringdown_fit(
            t0=t0, T=T, Mf=Mf, chif=chif, modes=modes, 
            mirror_modes=mirror_modes, hlm_modes=hlm_modes)

        # Append the mismatch to the list
        mm_list.append(bestfit['mismatch'])

    # Convert the list of mismatches to a grid
    mm_grid = np.reshape(
        np.array(mm_list), (len(Mf_array), len(chif_array)))
    
    return mm_grid


def plot_mismatch_M_chi_grid(self, outfile=None, plot_bestfit=False, 
                             fig_kw={}):
    """
    Plot the mismatch as a function of final mass and spin (from the last 
    call of mismatch_M_chi_grid) as a heatmap with a colourbar. 

    Parameters
    ----------
    outfile : str, optional
        File name to save the figure. If None, the figure is not saved. 
        The default is None.
        
    plot_bestfit : bool, optional
        Indicate the bestfit (minimum mismatch) mass-spin combination
        on the figure. If available, the variables stored to 
        self.Mf_bestfit and self.chif_bestfit are used. Otherwise, the
        minimum value in the grid is used. The default is False.
        
    fig_kw : dict, optional
        Additional keyword arguments to pass to plt.subplots() at the 
        figure creation.
    """
    # Display the mismatch grid as a colour map
    fig, ax = plt.subplots(**fig_kw)
    
    Mf_min = self.Mf_array[0]
    Mf_max = self.Mf_array[-1]
    chif_min = self.chif_array[0]
    chif_max = self.chif_array[-1]

    im = ax.imshow(
        np.log10(self.mm_grid), 
        extent=[chif_min,chif_max,Mf_min,Mf_max],
        aspect='auto',
        origin='lower',
        interpolation='bicubic',
        cmap='gist_heat_r')

    # Indicate true values
    ax.axhline(self.Mf, color='white', alpha=0.3)
    ax.axvline(self.chif_mag, color='white', alpha=0.3)
    
    # Indicate the bestfit values
    if plot_bestfit:
        
        # Find the bestfit mass and spin values, if these haven't already
        # been calculated
        if not hasattr(self, 'Mf_bestfit'):
            # Identify the index of the minimum mismatch value in the grid
            ind = np.unravel_index(
                np.argmin(self.mm_grid), self.mm_grid.shape)
                
            # Get the mass and spin values this corresonds to
            self.Mf_bestfit = self.Mf_array[ind[0]]
            self.chif_bestfit = self.chif_array[ind[1]]

        ax.plot(
            self.chif_bestfit, self.Mf_bestfit, marker='o', markersize=3, 
            color='black')
    
    # Colour bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('$\mathrm{log}_{10}\ \mathcal{M}$')

    ax.set_xlabel('$\chi_f$')
    ax.set_ylabel('$M_f\ [M]$')
    
    plt.tight_layout()
    
    if outfile is not None:
        plt.savefig(outfile)


def calculate_epsilon(self, t0=0, T=100, 
                      modes=[(2,2,n) for n in range(8)], mirror_modes=[],
                      hlm_modes=[(2,2)], method=None, x0=None):
    r"""
    Find the Mf and chif values that minimize the mismatch for a given 
    ringdown start time and model (these are stored as .Mf_bestfit and 
    .chif_bestfit), and from this calculate the 'distance' of the best fit 
    mass and spin values from the true remnant properties (expressed 
    through epsilon).

    Parameters
    ----------
    t0 : float
        The model start time to compute the mismatch for (relative to the 
        zero time).
        
    T : float, optional
        The end time of the analysis, relative to t0. The default is 100.
        
    modes : list, optional
        A list of (l,m,n) tuples (where l is the angular number of the 
        mode, m is the azimuthal number, and n is the overtone number) 
        specifying which modes to use in the model. The default is 
        [(2,2,n) for n in range(8)].
        
    mirror_modes : list, optional
        A list of (l,m,n) tuples (as in modes), specifying which 'mirror
        modes' to use in the model. The default is [] (no mirror modes are
        included).
        
    hlm_modes : list, optional
        A list of (l,m) tuples to specify which spherical harmonic hlm 
        modes the analysis should be performed on. The default is [(2,2)].
        
    method : str, optional
        The method used to find the mismatch minimum in the mass-spin
        space. Available methods are:
        
        - Any method available to scipy.optimize.minimize
            In this case, the scipy minimize function is called with the 
            given method. This includes None, in which case the method is 
            automatically chosen.
          
        - 'grid'
            The minimum mismatch is determined from the last call of
            mismatch_M_chi_grid. This is more reliable for models with
            large number of parameters, but the accuracy is determined by
            the choice of mass-spin limits and grid resolution.

    Returns
    -------
    epsilon : float
        The combined error between the bestfit Mf and chif values and the 
        true values. Defined as 
        
        .. math::
            \epsilon = \sqrt{ \left( \frac{\delta M_f}{M} \right)^2 + 
                              \left( \delta\chi_f \right)^2 }.
        
    delta_Mf : float
        The error between the bestfit Mf and true Mf values, defined as 
        
         .. math::
             \delta M_f = M_{\mathrm{best fit}} - M_f.
             
    delta_chif : float
        The error between the bestfit chif and true chif values, defined as
        
         .. math::
             \delta \chi_f = \chi_{\mathrm{best fit}} - \chi_f.
    """
    if method != 'grid':
        
        if x0 is None:
            # Take the true final mass and spin values as the initial guess
            x0 = [self.Mf, self.chif_mag]
        
        def mismatch_M_chi(x, t0, T, modes, model, hlm_modes):
            """
            A wrapper for the ringdown_fit function, for use with the 
            SciPy minimize function.
            """
            # Get the mass and spin values from the x list
            Mf = x[0]
            chif = x[1]
            
            if chif > 0.99:
                chif = 0.99
            if chif < 0:
                chif = 0
            
            # Run the fit
            bestfit = self.ringdown_fit(
                t0=t0, T=T, Mf=Mf, chif=chif, modes=modes, 
                mirror_modes=mirror_modes, hlm_modes=hlm_modes)
            
            # Store useful quantities
            self.mm_bestfit = bestfit['mismatch']
            self.C_bestfit = bestfit['C']
            
            return bestfit['mismatch']

        # Call the SciPy minimization
        res = minimize(
            mismatch_M_chi, x0,
            args=(t0, T, modes, mirror_modes, hlm_modes),
            method=method, bounds=[(0,1.5), (0,0.99)], 
            options={'xatol':1e-6,'disp':False})

        # Extract best fit parameters
        self.Mf_bestfit = res.x[0]
        self.chif_bestfit = res.x[1]
        
    elif method == 'grid':
            
        # Identify the index of the minimum mismatch value in the grid
        ind = np.unravel_index(
            np.argmin(self.mm_grid), self.mm_grid.shape)
            
        # Get the mass and spin values this corresonds to
        self.Mf_bestfit = self.Mf_array[ind[0]]
        self.chif_bestfit = self.chif_array[ind[1]]
        
    # Calculate differences to the true values (following the 
    # convention in arXiv:1903.08284)
    delta_Mf = self.Mf_bestfit - self.Mf
    delta_chif = self.chif_bestfit - self.chif_mag
    epsilon = np.sqrt((delta_Mf/(self.m1+self.m2))**2 + delta_chif**2)
        
    return epsilon, delta_Mf, delta_chif