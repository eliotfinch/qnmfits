import numpy as np


def mismatch(times, wf_1, wf_2):
    """
    Calculates the mismatch (see, for example, arXiv:1903.08284) between two
    complex waveforms.

    Parameters
    ----------
    times : array
        The times at which the waveforms are evaluated.
        
    wf_1, wf_2 : array
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


def sky_averaged_mismatch(times, wf_dict_1, wf_dict_2):
    """
    Calculates the sky averaged mismatch between two dictionaries of spherical
    harmonic waveform modes. 
    
    If the two dictionaries have a different set of keys, the sum is performed
    over the keys of wf_dict_1 (this may be the case, for example, if only a 
    subset of spherical harmonic modes are modelled).

    Parameters
    ----------
    times : TYPE
        DESCRIPTION.
        
    wf_dict_1 : TYPE
        DESCRIPTION.
        
    wf_dict_2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
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
