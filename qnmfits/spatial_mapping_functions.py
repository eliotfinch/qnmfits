"""
These functions are used to perform the spatial mapping of QNMs and QQNMs,
and to make predictions of the spatial structure of QQNMs. They were 
written by Richard Dyer to be used with the repo 
https://github.com/Richardvnd/spatial_mapping. 

"""

import numpy as np
import qnmfits
import quaternionic
import spherical
import spheroidal
from scipy.integrate import dblquad as dbl_integrate
from spherical import Wigner3j as w3j


def mapping_multimode_ringdown_fit(
    times,
    data_dict,
    modes,
    Mf,
    chif,
    t0,
    mapping_modes,
    t0_method="geq",
    T=100,
    spherical_modes=None,
):
    """
    Performs a spatial mapping of a QNM or QQNM. Note that any QQNMs included in the model
    which are not being spatially mapped will be fit using the B (i) mode mixing prediction.

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

    mapping_modes : array_like
        A sequence of QNM or QQNM tuples to specify which modes spatially map.

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

    frequencies: array_like, optional
        A sequence of complex frequencies to use in the analysis. If None,
        the frequencies are calculated using the qnm module. The default is
        None.

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
                amplitude for each ringdown mode, in addition to one amplitude
                of the spatially mapped modes for each spherical mode.
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
    if t0_method == "geq":

        data_mask = (times >= t0) & (times < t0 + T)

        times = times[data_mask]
        data = np.concatenate([data_dict[lm][data_mask] for lm in spherical_modes])
        data_dict_mask = {lm: data_dict[lm][data_mask] for lm in spherical_modes}

    elif t0_method == "closest":

        start_index = np.argmin((times - t0) ** 2)
        end_index = np.argmin((times - t0 - T) ** 2)

        times = times[start_index:end_index]
        data = np.concatenate(
            [data_dict[lm][start_index:end_index] for lm in spherical_modes]
        )
        data_dict_mask = {
            lm: data_dict[lm][start_index:end_index] for lm in spherical_modes
        }

    else:
        print(
            """Requested t0_method is not valid. Please choose between
              'geq' and 'closest'."""
        )

    data_dict = data_dict_mask

    mod_modes = modes.copy()
    for mapping_mode in mapping_modes:
        if mapping_mode in mod_modes:
            mod_modes.remove(mapping_mode)
        else:
            modes.append(mapping_mode)

    linear_modes = []
    quadratic_modes = []

    for mode in mod_modes:
        if len(mode) == 4:
            linear_modes.append(mode)
        elif len(mode) == 8:
            quadratic_modes.append(mode)
        else:
            raise ValueError(f"Wrong number of indices in tuple: {mode}.")

    mod_modes = linear_modes + quadratic_modes

    # Construct the coefficient matrix for use with NumPy's lstsq function.

    # Mixing coefficients
    # -------------------

    # A list of lists for the mixing coefficient indices. The first list is
    # associated with the first lm mode. The second list is associated with
    # the second lm mode, and so on.
    # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')],
    #        [(3,2,2',2',0'), (3,2,3',2',0')] ]

    indices_lists_l = [
        [lm_mode + mode for mode in linear_modes] for lm_mode in spherical_modes
    ]

    mu_lists = [qnmfits.qnm.mu_list(indices, chif) for indices in indices_lists_l]

    indices_lists_q = [
        [lm_mode + mode for mode in quadratic_modes] for lm_mode in spherical_modes
    ]

    alpha_lists = [
        Qmu_B(indices, chif, l_max=8, s1=-2, s=0) for indices in indices_lists_q
    ]

    coef_lists = [mu + alpha for mu, alpha in zip(mu_lists, alpha_lists)]

    identity = np.eye(len(spherical_modes))
    identitys = np.hstack([identity] * len(mapping_modes))
    coef_lists = [list + i_row.tolist() for list, i_row in zip(coef_lists, identitys)]
    mod_modes += [
        mapping_mode
        for mapping_mode in mapping_modes
        for _ in range(len(spherical_modes))
    ]

    # Frequencies
    # ------------

    frequencies = np.array(qnmfits.qnm.omega_list(mod_modes, chif, Mf))

    # Construct coefficient matrix and solve
    # --------------------------------------

    # Construct the coefficient matrix
    a = np.concatenate(
        [
            np.array(
                [
                    coef_lists[i][j] * np.exp(-1j * frequencies[j] * (times - t0))
                    for j in range(len(frequencies))
                ]
            ).T
            for i in range(len(spherical_modes))
        ]
    )

    # Solve for the complex amplitudes, C. Also returns the sum of
    # residuals, the rank of a, and singular values of a.
    C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)

    # Evaluate the model. This needs to be split up into the separate
    # spherical harmonic modes.
    model = np.einsum("ij,j->i", a, C)

    # Split up the result into the separate spherical harmonic modes, and
    # store to a dictionary. We also store the "weighted" complex amplitudes
    # to a dictionary.
    model_dict = {}
    weighted_C = {}

    for i, lm in enumerate(spherical_modes):
        model_dict[lm] = model[i * len(times) : (i + 1) * len(times)]
        weighted_C[lm] = np.array(coef_lists[i]) * C

    # Calculate the (sky-averaged) mismatch for the fit
    mm = qnmfits.multimode_mismatch(times, model_dict, data_dict)

    # Create a list of mode labels (can be used for plotting)
    labels = [str(mode) for mode in mod_modes]

    # Store all useful information to a output dictionary
    best_fit = {
        "residual": res,
        "mismatch": mm,
        "C": C,
        "weighted_C": weighted_C,
        "data": data_dict,
        "model": model_dict,
        "model_times": times,
        "spherical_modes": spherical_modes,
        "t0": t0,
        "modes": mod_modes,
        "mode_labels": labels,
        "frequencies": frequencies,
    }

    # Return the output dictionary
    return best_fit


def spatial_reconstruction(theta, phi, best_fit, map, l_max, s3=-2):
    """
    A function to reconstruct a QNM or QQNM from the best-fit complex amplitudes.

    Parameters
    ----------
    theta : array_like
        The polar angles at which to evaluate the reconstructed waveform.
    phi : array_like
        The azimuthal angles at which to evaluate the reconstructed waveform.
    best_fit : dict
        The output dictionary from the mapping_multimode_ringdown_fit function.
    map : tuple
        The indices of the mode to be reconstructed.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    s3 : int, optional
        The spin weight of the spherical harmonic to use in the reconstruction.

    Returns
    -------
    ans : array_like
        The (normalised) reconstructed waveform.

    """

    wigner = spherical.Wigner(l_max)
    R = quaternionic.array.from_spherical_coordinates(theta, phi)
    Y = wigner.sYlm(s3, R)

    mask = np.array([mode == map for mode in best_fit["modes"]])
    ans = sum(
        A * Y[:, :, wigner.Yindex(lp, mp)]
        for (lp, mp), A in zip(best_fit["spherical_modes"], best_fit["C"][mask])
    )
    ans /= np.max(np.abs(ans))

    return ans


def spatial_prediction_linear(theta, phi, map, l_max, chif):
    """
    A function to predict the spatial distribution of a QNM using
    the linear mode mixing coefficients taken from first order PT.

    Parameters
    ----------
    theta : array_like
        The polar angles at which to evaluate the reconstructed waveform.
    phi : array_like
        The azimuthal angles at which to evaluate the reconstructed waveform.
    map : tuple
        The indices of the mode to be reconstructed.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    chif : float
        The magnitude of the remnant black hole spin.

    Returns
    -------
    ans : array_like
        The predicted (normalised) spatial distribution of the QNM.

    """
    wigner = spherical.Wigner(l_max)
    R = quaternionic.array.from_spherical_coordinates(theta, phi)
    Y = wigner.sYlm(-2, R)

    l, m, n, p = map
    ans = sum(
        qnmfits.qnm.mu(lp, m, l, m, n, p, chif) * Y[:, :, wigner.Yindex(lp, m)]
        for lp in np.arange(2, l_max + 1)
    )

    ans /= np.max(np.abs(ans))
    return ans


def spatial_prediction_quadratic(theta, phi, map, l_max, chif, Qmu, **kwargs):
    """
    A function to predict the spatial distribution of a QQNM using the quadratic
    mode mixing coefficients predicted by second order PT.

    Parameters
    ----------
    theta : array_like
        The polar angles at which to evaluate the reconstructed waveform.
    phi : array_like
        The azimuthal angles at which to evaluate the reconstructed waveform.
    map : tuple
        The indices of the mode to be reconstructed.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    chif : float
        The magnitude of the remnant black hole spin.
    Qmu : function
        The choice of prediction to calculate the quadratic mode mixing coefficients.
    **kwargs :
        Spin weight arguments to pass to the Qmu function.

    Returns
    -------
    ans : array_like
        The predicted (normalised) spatial distribution of the QQNM.

    """

    s1 = kwargs.get("s1", -2)
    s2 = kwargs.get("s2", 0)
    s3 = kwargs.get("s3", -2)

    wigner = spherical.Wigner(l_max)
    R = quaternionic.array.from_spherical_coordinates(theta, phi)
    Y = wigner.sYlm(s3, R)

    a, b, c, sign1, e, f, g, sign2 = map
    j = b + f

    lpp = max(abs(j), abs(s3))
    ans = sum(
        Qmu([(i, j) + map], chif, l_max, s1=s1, s2=s2)[0] * Y[:, :, wigner.Yindex(i, j)]
        for i in np.arange(lpp, l_max + 1)
    )

    ans /= np.max(np.abs(ans))
    return ans


def spatial_prediction_C(theta, phi, map, chif):
    """
    A faster method to predict the spatial distribution of a QQNM using the
    C prediction for the QQNM spatial structure (which avoids computing mode
    mixing coefficients).

    Parameters
    ----------
    theta : array_like
        The polar angles at which to evaluate the reconstructed waveform.
    phi : array_like
        The azimuthal angles at which to evaluate the reconstructed waveform.
    map : tuple
        The indices of the mode to be reconstructed.
    chif : float
        The magnitude of the remnant black hole spin.

    Returns
    -------
    ans : array_like
        The predicted (normalised) spatial distribution of the QQNM.

    """

    a, b, c, sign1, e, f, g, sign2 = map
    L = a + e
    j = b + f

    omega = qnmfits.qnm.omega_list([(a, b, c, sign1, e, f, g, sign2)], chif, 1)
    gamma = chif * omega[0]

    ans = spheroidal.harmonic(-2, L, j, gamma)(theta, phi)

    ans /= np.max(np.abs(ans))

    return ans


def spatial_mismatch_linear(best_fit, map, chif, l_max=8):
    """
    The spatial mismatch between the best-fit waveform and the predicted spatial
    distribution of a QNM using the linear mode mixing coefficients.

    Parameters
    ----------
    best_fit : dict
        The output dictionary from the mapping_multimode_ringdown_fit function.
    map : tuple
        The indices of the mode to be reconstructed.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int, optional
        The maximum l-mode to consider in the reconstruction. The default is 8.

    Returns
    -------
    sm : float
        The spatial mismatch between the best-fit waveform and the predicted
        spatial distribution of the QNM.
    arg : float
        The phase difference between the best-fit waveform and the predicted
        spatial distribution of the QNM.
    z : complex
        The complex inner product between the best-fit waveform and the predicted
        spatial distribution

    """

    mask = np.array([mode == map for mode in best_fit["modes"]])

    l, m, n, p = map
    z = sum(
        A * np.conj(qnmfits.qnm.mu(lp, mp, l, m, n, p, chif))
        for (lp, mp), A in zip(best_fit["spherical_modes"], best_fit["C"][mask])
    )
    denominator2 = np.abs(
        sum(
            qnmfits.qnm.mu(lp, m, l, m, n, p, chif)
            * np.conj(qnmfits.qnm.mu(lp, m, l, m, n, p, chif))
            for lp in range(2, l_max + 1)
        )
    )
    numerator = np.abs(z)
    denominator1 = np.abs(sum(best_fit["C"][mask] * np.conj(best_fit["C"][mask])))
    denominator = np.sqrt(denominator1 * denominator2)
    arg = np.angle(z)
    sm = 1 - numerator / denominator

    return sm, arg, z


def spatial_mismatch_quadratic(best_fit, map, l_max, chif, Qmu, **kwargs):
    """
    The spatial mismatch between the best-fit waveform and the predicted spatial
    distribution of a QQNM using the quadratic mode mixing coefficients.

    Parameters
    ----------
    best_fit : dict
        The output dictionary from the mapping_multimode_ringdown_fit function.
    map : tuple
        The indices of the mode to be reconstructed.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    chif : float
        The magnitude of the remnant black hole spin.
    Qmu : function
        The choice of prediction to calculate the quadratic mode mixing coefficients.
    **kwargs :
        Spin weight arguments to pass to the Qmu function.

    Returns
    -------
    sm : float
        The spatial mismatch between the best-fit waveform and the predicted
        spatial distribution of the QQNM.
    arg : float
        The phase difference between the best-fit waveform and the predicted
        spatial distribution of the QQNM.
    z : complex
        The complex inner product between the best-fit waveform and the predicted
        spatial distribution

    """

    s1 = kwargs.get("s1", -2)
    s2 = kwargs.get("s2", 0)

    a, b, c, sign1, e, f, g, sign2 = map
    j = b + f

    mask = np.array([mode == map for mode in best_fit["modes"]])

    z = sum(
        A * np.conj(Qmu([(lp, mp) + map], chif, l_max, s1=s1, s2=s2)[0])
        for (lp, mp), A in zip(best_fit["spherical_modes"], best_fit["C"][mask])
    )
    denominator2 = np.abs(
        sum(
            Qmu([(lp, j) + map], chif, l_max, s1=s1, s2=s2)[0]
            * np.conj(Qmu([(lp, j) + map], chif, l_max, s1=s1, s2=s2)[0])
            for lp in range(2, l_max + 1)
        )
    )
    numerator = np.abs(z)
    denominator1 = np.abs(sum(best_fit["C"][mask] * np.conj(best_fit["C"][mask])))
    denominator = np.sqrt(denominator1 * denominator2)
    arg = np.angle(z)
    sm = 1 - numerator / denominator

    return sm, arg, z


def spatial_data_mismatch(best_fit1, best_fit2, map):
    """
    The spatial mismatch between two best-fit waveforms; used
    to determine the numerical error in the NR data.

    Parameters
    ----------
    best_fit1 : dict
        An output dictionary from the mapping_multimode_ringdown_fit function.
    best_fit2 : dict
        An output dictionary from the mapping_multimode_ringdown_fit function.
    map : tuple
        The indices of the mode to be considered.

    Returns
    -------
    sm : float
        The spatial mismatch between the two best-fit waveforms.

    """

    mask = np.array([mode == map for mode in best_fit1["modes"]])

    numerator = np.abs(sum(best_fit1["C"][mask] * np.conj(best_fit2["C"][mask])))
    denominator1 = np.abs(sum(best_fit1["C"][mask] * np.conj(best_fit1["C"][mask])))
    denominator2 = np.abs(sum(best_fit2["C"][mask] * np.conj(best_fit2["C"][mask])))
    denominator = np.sqrt(denominator1 * denominator2)

    return 1 - numerator / denominator


def data_mismatch(sim1, sim2, t0=0, modes=None, T=100, dt=0.01, shift=0):
    """
    The mismatch between two simulation level and extraction radii; used
    to determine the numerical error in the NR data.

    Parameters
    ----------
    sim1 : dict
        The output dictionary from the first simulation data level and radius.
    sim2 : dict
        The output dictionary from the second simulation data level and radius.
    t0 : float, optional
        The start time of the integration. The default is 0.
    modes : array_like, optional
        A sequence of (l,m) tuples to specify which spherical modes to include in
        the ringdown model. The default is None, which reverts to including all modes.
    T : float, optional
        The duration integrated over, such that the end time is t0 + T.
        The default is 100.
    dt : float, optional
        The step size of the integration. The default is 0.01.
    shift : float, optional
        The time shift between the two simulations. The default is 0 (the workbooks
        typically perform the shift before passing to this function).

    Returns
    -------
    mismatch : float
        The mismatch between the two simulations.
    """

    new_times = np.arange(t0, t0 + T, dt)

    if modes is None:
        modes = list(sim1.h.keys())

    numerator = 0.0
    denominator1 = 0.0
    denominator2 = 0.0

    for mode in modes:

        h1 = sim1.h[mode]
        h2 = sim2.h[mode]

        interp_h1 = np.interp(new_times, sim1.times, h1)
        interp_h2 = np.interp(new_times - shift, sim2.times, h2)

        numerator += np.abs(np.trapz(interp_h1 * np.conjugate(interp_h2), x=new_times))
        denominator1 += np.abs(
            np.trapz(interp_h1 * np.conjugate(interp_h1), x=new_times)
        )
        denominator2 += np.abs(
            np.trapz(interp_h2 * np.conjugate(interp_h2), x=new_times)
        )

    denominator = np.sqrt(denominator1 * denominator2)

    return 1 - (numerator / denominator)


def sYlm(l, m, theta, phi, s=-2, l_max=8):
    """
    A function to calculate the spin-weighted spherical harmonics.

    Parameters
    ----------
    l : int
        The l-mode of the spherical harmonic.
    m : int
        The m-mode of the spherical harmonic.
    theta : array_like
        The polar angles at which to evaluate the spherical harmonic.
    phi : array_like
        The azimuthal angles at which to evaluate the spherical harmonic.
    s : int, optional
        The spin weight of the spherical harmonic. The default is -2.
    l_max : int, optional
        The maximum l-mode to consider in the reconstruction. The default is 8.

    Returns
    -------
    Y : array_like
        The spin-weighted spherical harmonic evaluated at the given angles.

    """
    wigner = spherical.Wigner(l_max)
    R = quaternionic.array.from_spherical_coordinates(theta, phi)
    Y = wigner.sYlm(s, R)
    return Y[wigner.Yindex(l, m)]


def kappa(i, j, d, h, b, f, s1, s2):
    """
    A function to determine the kappa coefficient in the quadratic mode mixing.

    Parameters
    ----------
    i : int
        The l-mode of the third spherical mode.
    j : int
        The m-mode of the third spherical mode.
    d : int
        The l-mode of the first spherical mode.
    h : int
        The l-mode of the second spherical mode.
    b : int
        The m-mode of the first spherical mode.
    f : int
        The m-mode of the second spherical mode.
    s1 : int
        The spin weight of the first spheroidal harmonic.
    s2 : int
        The spin weight of the second spheroidal harmonic.

    Returns
    -------
    kappa : float
        The kappa coefficient.

    """

    return (
        (((2 * d + 1) * (2 * h + 1) * (2 * i + 1)) / (4 * np.pi)) ** (1 / 2)
        * w3j(d, h, i, -s1, -s2, s1 + s2)
        * w3j(d, h, i, b, f, -j)
        * (-1) ** (j + s1 + s2)
    )


def Qmu_A(indices, chif, l_max, **kwargs):
    """

    A function to calculate the A prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the A prediction for.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    **kwargs :
        Here for consistency when passed into other functions.

    Returns
    -------
    Qmu : array_like
        The quadratic mode mixing coefficients. 

    """

    return [
        sum(
            qnmfits.qnm.mu(d, b, a, b, c, sign1, chif, -2)
            * qnmfits.qnm.mu(h, f, e, f, g, sign2, chif, -2)
            * kappa(i, j, d, h, b, f, -2, -2)
            for d in range(2, l_max + 1)
            for h in range(2, l_max + 1)
        )
        for i, j, a, b, c, sign1, e, f, g, sign2 in indices
    ]


def Qmu_B(indices, chif, l_max, **kwargs):
    """
    A function to calculate the B prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the B prediction for.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    **kwargs :
        Spin weight arguments to pass to the Qmu function.

    Returns
    -------
    Qmu : array_like
        The quadratic mode mixing coefficients. 

    """

    s1 = kwargs.get("s1", -2)
    s2 = kwargs.get("s2", 0)

    return [
        sum(
            qnmfits.qnm.mu(d, b, a, b, c, sign1, chif, s1)
            * qnmfits.qnm.mu(h, f, e, f, g, sign2, chif, s2)
            * kappa(i, j, d, h, b, f, s1, s2)
            for d in range(np.abs(s1), l_max + 1)
            for h in range(np.abs(s2), l_max + 1)
        )
        for i, j, a, b, c, sign1, e, f, g, sign2 in indices
    ]


def Qmu_C(indices, chif, l_max, **kwargs):
    """
    A function to calculate the C prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the C prediction for.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    **kwargs :
        Here for consistency when passed into other functions.

    Returns
    -------
    Qmu : array_like
        The quadratic mode mixing coefficients. 

    """

    alphas = []

    for i, j, a, b, c, sign1, e, f, g, sign2 in indices:
        L = a + e
        M = b + f
        omega = qnmfits.qnm.omega_list([(a, b, c, sign1, e, f, g, sign2)], chif, 1)
        gamma = chif * omega[0]
        S = spheroidal.harmonic(-2, L, M, gamma)

        def f_real(theta, phi):
            return np.real(
                np.sin(theta) * S(theta, phi) * np.conj(sYlm(i, j, theta, phi))
            )

        def f_imag(theta, phi):
            return np.imag(
                np.sin(theta) * S(theta, phi) * np.conj(sYlm(i, j, theta, phi))
            )

        alpha_real = dbl_integrate(f_real, 0, 2 * np.pi, 0, np.pi)[0]
        alpha_imag = dbl_integrate(f_imag, 0, 2 * np.pi, 0, np.pi)[0]

        alphas.append(alpha_real + 1j * alpha_imag)

    return alphas


def Qmu_D(indices, chif, l_max, **kwargs):
    """
    A function to calculate the D prediction for the QQNM mode mixing.

    Parameters
    ----------
    indices : array_like
        A sequence of tuples to specify which spherical mode QQNM combinations to
        calculate the D prediction for.
    chif : float
        The magnitude of the remnant black hole spin.
    l_max : int
        The maximum l-mode to consider in the reconstruction.
    **kwargs :
        Here for consistency when passed into other functions.

    Returns
    -------
    Qmu : array_like
        The quadratic mode mixing coefficients. 

    """

    return [
        sum(
            qnmfits.qnm.mu(d, b, a, b, c, sign1, chif, -2)
            * qnmfits.qnm.mu(h, f, e, f, g, sign2, chif, -2)
            * kappa(i, j, d, h, b, f, -2, -2)
            * np.sqrt((i + 4) * (i - 3) * (i + 3) * (i - 2))
            for d in range(2, l_max + 1)
            for h in range(2, l_max + 1)
        )
        for i, j, a, b, c, sign1, e, f, g, sign2 in indices
    ]
