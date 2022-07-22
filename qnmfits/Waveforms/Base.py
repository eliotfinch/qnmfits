import numpy as np
import matplotlib.pyplot as plt

import quaternionic

from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from tqdm import tqdm

# Mismatch functions
from ..utils import sky_averaged_mismatch

# Class to load QNM frequencies and mixing coefficients
from ..qnm import qnm
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
    time : array
        The times at which the model is evalulated.
        
    start_time : float
        The time at which the model begins. Should lie within the times array.
        
    complex_amplitudes : list
        A list of complex amplitudes.
        
    frequencies : list
        The complex frequencies of the modes. These should be ordered in the
        same order as the amplitudes.

    Returns
    -------
    h : array
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


class BaseClass:
    """
    A base class which contains all the methods used by the other waveform
    classes
    """
    
    def create_interpolant(self):
        """
        Use a spline to interpolate the strain data, ready to evaluate on a 
        chosen array of times. The dictionary of functions are stored to
        self.h_interp.
        """
        self.h_interp_funcs = {}
        
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):
                
                real_h_interp = spline(self.times, self.h[l,m].real, ext=3)
                imag_h_interp = spline(self.times, self.h[l,m].imag, ext=3)
                
                self.h_interp_funcs[l,m] = (real_h_interp, imag_h_interp)
                    
        self.h_interp = lambda l, m, t: self.h_interp_funcs[l,m][0](t) \
            + 1j*self.h_interp_funcs[l,m][1](t)
    
    def uniform_times(self):
        """
        Interpolate h and evaluate on an array of times which are uniformly 
        spaced. The spacing is chosen to be the minimum spacing in the default
        time array.
        """
        # New time spacing and time array
        dt = np.min(np.diff(self.times))
        uniform_times = np.arange(self.times[0], self.times[-1], dt)
        
        # Dictionary to temporarily store the new modes
        hp = {}
        
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):
                
                # Load the mode data
                data = self.h[l,m]
                
                # Interpolate with a spline, and evaulate on the uniform times
                hp[l,m] = \
                    spline(self.times, data.real)(uniform_times) \
                    + 1j*spline(self.times, data.imag)(uniform_times)
                    
        # Overwrite times and the h dictionary
        self.times = uniform_times
        self.h = hp
        
    
    def rotate_modes(self):
        """
        Reperform the spin-weighted spherical harmonic decomposition in a 
        rotated coordinate system where the z-axis is parallel to the remnant 
        black hole spin.
        
        The modes in the h dictionary are over-written.
        """
        # Get the quaternion representing the rotation needed for the basis
        
        # We use a rotation vector (in axis-angle representation) to perform
        # the rotation about a single axis. This preserves the phase of the
        # spherical modes.
        rot = np.cross([0,0,1], self.chif)
        
        # Scale the magnitude of the rotation vector to be thetaf
        rot = self.thetaf*rot/np.linalg.norm(rot)
        
        # Quaternion representation
        R = quaternionic.array.from_axis_angle(rot)
        
        # And the Wigner D matrix
        D = self.wigner.D(R)
        
        # Dictionary to temporarily store the new modes
        hp = {}
        
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):
                
                hp[l,m] = np.zeros_like(self.h[l,m])
                
                # Construct the new modes
                for mp in range(-l,l+1):
                    hp[l,m] += D[self.wigner.Dindex(l, mp, m)]*self.h[l,mp]
            
        # Overwrite the h dictionary
        self.h = hp
        
        # The remnant black hole spin is now along the z axis
        self.chif = np.array([0,0,self.chif_mag])
        
        
    def rotate_modes_over_time(self):
        """
        Reperform the spin-weighted spherical harmonic decomposition in a 
        rotated coordinate system where the z-axis is parallel to the 
        instantanious black hole spin.
        
        The modes in the h dictionary are over-written.
        """
        # Reshape chioft magnitudes to make division possible
        chioft_mag = self.chioft_mag.reshape(len(self.chioft_mag),1)
        
        # Get the changing spin direction
        chioft_norm = self.chioft/chioft_mag
        thetaoft = np.arccos(chioft_norm[:,2])
        phioft = np.arctan2(chioft_norm[:,1], chioft_norm[:,0])
        
        print(f'Time length = {len(chioft_norm)}')
        
        # Get the quaternion representing the rotation needed for the basis
        Roft = quaternionic.array.from_spherical_coordinates(thetaoft, phioft)
        
        # And the Wigner D matrix
        Doft = self.wigner.D(Roft)
        
        # Dictionary to temporarily store the new modes
        hp = {}
        
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):
                
                hp[l,m] = np.zeros_like(self.h[l,m])
                    
                # Construct the new modes
                for mp in range(-l,l+1):
                    hp[l,m] += Doft[:,self.wigner.Dindex(l, mp, m)]*self.h[l,mp]
            
        # Overwrite the h dictionary
        self.h = hp
        
        
    def calculate_hdot(self):
        '''
        Caclulate the derivative of the strain (the time array can be the 
        default, non-uniformly spaced array). First, interpolate h with a 
        standard spline, then differentiate the spline and evaluate that 
        derivative at self.times. Results are stored to the hdot dictionary.
        '''
        # Dictionary to store the mode derivatives
        self.hdot = {}
        
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):

                # Load the mode data
                data = self.h[l,m]
                
                # Calculate the derivative with splines and store to dictionary
                self.hdot[l,m] = \
                    spline(self.times, data.real).derivative()(self.times) \
                    + 1j*spline(self.times, data.imag).derivative()(self.times)

        
    def hdot_lm(self, l, m):   
        """
        Correct the hdot values to return zero if either l or m are not 
        allowed (this is how the expressions of arXiv:0707.4654 are supposed 
        to be used). Returns a single mode.
        """
        if l<2 or l>self.ellMax:
            return np.zeros_like(self.times, dtype=complex)
        elif m<-l or m>l:
            return np.zeros_like(self.times, dtype=complex)
        else:
            return self.hdot[l,m]
        
        
    def calculate_Moft(self):
        """
        Calculate the radiated energy as a function of time, using Eq. (3.8) 
        of arXiv:0707.4654, then use this to compute the evolving mass of the 
        remnant black hole.
        """
        # Calculate dEdt
        self.Edot = np.zeros_like(self.times)
        
        for l in range(2, self.ellMax+1):
            for m in range(-l, l+1):
                self.Edot += (1/(16*np.pi)) * abs(self.hdot[l,m])**2
                
        # As we are interested in how the mass of the final black hole evolves
        # from its final value Mf (=Ef), we integrate Edot backwards from the 
        # known value of Mf. To do this, first construct a spline of Edot, for
        # which we can find its integral (antiderivative).
        Eint = spline(self.times, self.Edot).antiderivative()
        
        # M(t) = Mf + int_t^T Edot(t') dt', where T is the final time. 
        self.Moft = self.Mf + (Eint(self.times[-1]) - Eint(self.times))
        
        
    def calculate_chioft(self):
        """
        Calculate the angular momentum flux (for the x, y and z components)
        using Eq. (3.22-3.24) of arXiv:0707.4654, then use this to compute the
        evolving final spin of the remnant black hole.
        """
        # Coefficient function
        def flm(l,m):
            return np.sqrt(l*(l+1) - m*(m+1))
        
        # Calculate each component of Jdot
        Jxdot = np.zeros_like(self.times)
        
        for l in range(2, self.ellMax+1):
            for m in range(-l, l+1):
                Jxdot += (1/(32*np.pi)) * np.imag(
                    self.h[l,m] * (flm(l,m)*np.conj(self.hdot_lm(l,m+1))
                                   + flm(l,-m)*np.conj(self.hdot_lm(l,m-1))))
                
        Jydot = np.zeros_like(self.times)
        
        for l in range(2, self.ellMax+1):
            for m in range(-l, l+1):
                Jydot += - (1/(32*np.pi)) * np.real(
                    self.h[l,m] * (flm(l,m)*np.conj(self.hdot_lm(l,m+1))
                                   - flm(l,-m)*np.conj(self.hdot_lm(l,m-1))))
        
        Jzdot = np.zeros_like(self.times)
        
        for l in range(2, self.ellMax+1):
            for m in range(-l, l+1):
                Jzdot += (1/(16*np.pi)) * np.imag(
                    m*self.h[l,m] * np.conj(self.hdot_lm(l,m)))
                
        # Combine the components
        self.Jdot = np.transpose([Jxdot, Jydot, Jzdot])
        
        # As with integrating the energy flux, we integrate the angular 
        # momentum backwards from the final known value of chif. Note, we need
        # to first convert between angular momentum J and the dimensionless 
        # spin vector (we reshape Moft to make division possible).
        chidot = self.Jdot/(self.Moft**2).reshape(len(self.Moft),1)
        
        # Initialize the list to store the evolution for each of chi's 
        # components
        chioft = []
        
        for i, component in enumerate(np.transpose(chidot)):
            
            # Construct a spline for each component of chidot, then calculate
            # its integral
            chiint = spline(self.times, component).antiderivative()
            
            # Integrate backwards from the final chi value
            chioft.append(
                self.chif[i] + (chiint(self.times[-1]) - chiint(self.times)))
        
        # Reshape and calculate norm at each time
        self.chioft = np.transpose(chioft)
        self.chioft_mag = np.linalg.norm(self.chioft, axis=1)
        
        
    def project_signal(self, theta, phi=None):
        """
        Project the signal in the requested position on the (source frame) sky.

        Parameters
        ----------
        theta : float
            The angle between the source-frame z-axis and the line of sight 
            (i.e. the inclination).
            
        phi : float, optional
            The azimuthal angle of the line of sight in the source frame. The 
            default is None.

        Returns
        -------
        signal : complex array
            The complex gravitational wave signal.
            
        """
        # Initialize an empty, complex array
        signal = np.zeros_like(self.times, dtype=complex)
        
        if phi is None:
            phi = 0 # self.phif
            
        # Get the quaternion representing the rotation needed
        R = quaternionic.array.from_spherical_coordinates(theta, phi)
        
        # And the spin-weighted spherical harmonic
        Y = self.wigner.sYlm(-2, R)
        
        # Compute the projected signal. This is done by summing each of the
        # modes, weighted by the spin-weighted spherical harmonics.
        for l in range(2, self.ellMax+1):
            for m in range(-l, l+1):
                signal += self.h[l,m] * Y[self.wigner.Yindex(l,m)]
                
        return signal
                
                
    # def project_twisted_signal(self):
    #     """
    #     Project the signal along the changing direction of the total angular 
    #     momentum vector (equivalent to the direction of chi(t)).
    #     """
    #     # Initialize an empty, complex array
    #     self.twisted_signal = np.zeros_like(self.times, dtype=complex)
        
    #     # Reshape chioft magnitudes to make division possible
    #     chioft_mag = self.chioft_mag.reshape(len(self.chioft_mag),1)
        
    #     # Angular coordinates of the spin vector
    #     chioft_norm = self.chioft/chioft_mag
    #     thetaoft = np.arccos(chioft_norm[:,2])
    #     phioft = np.arctan2(chioft_norm[:,1], chioft_norm[:,0])
        
    #     if self.hlm_prime:
    #         # If the hlm modes have been rotated, we need to be careful with
    #         # the argument of the sYlm. Will subtracting thetaf and phif fix
    #         # this?
    #         thetaoft -= self.thetaf
    #         phioft -= self.phif

    #     # Compute the projected signal. This is done by summing each of the
    #     # modes, weighted by the spherical harmonics (now for each time step).
    #     for i in range(len(self.times)):
    #         for l in range(2, self.ellMax+1):
    #             for m in range(-l, l+1):
    #                 self.twisted_signal[i] += \
    #                     self.h[l,m][i] * sYlm(-2, l, m, thetaoft[i], phioft[i])
                        
    def time_shift(self):
        """
        Shift the time array so that t=0 is defined by the requested method.
        """
        if type(self.zero_time) is float:
            if self.zero_time == 0:
                self.zero_time_method = 'Simulation default'
            else:
                self.zero_time_method = 'User defined'
            
        elif type(self.zero_time) is tuple:
            self.zero_time_method = f'{self.zero_time} peak'
            # Ampltude of the requested mode
            amp = abs(self.h[self.zero_time])
            # We choose t=0 to correspond to the peak of the mode
            self.zero_time = self.times[np.argmax(amp)]
            
        elif self.zero_time == 'norm':
            self.zero_time_method = 'Norm peak'
            # All hlm modes up to ellMax
            all_modes = [
                (l,m) for l in range(2,self.ellMax+1) for m in range(-l,l+1)]
            stacked_strain = np.vstack([self.h[lm] for lm in all_modes])
            # Total amplitude of all the available modes
            amp = np.sqrt(np.sum(abs(stacked_strain)**2, axis=0))
            # We choose t=0 to correspond to the peak of this total amplitude
            self.zero_time = self.times[np.argmax(amp)]
            
        elif self.zero_time == 'Edot':
            self.zero_time_method = 'Edot peak'
            self.zero_time = self.times[np.argmax(self.Edot)]
            
        elif self.zero_time == 'common_horizon':
            self.zero_time_method = 'Common horizon'
            self.zero_time = self.common_horizon_time
            
        self.times = self.times - self.zero_time
                        
                        
    def calculate_foft(self, method='phase_derivative'):
        """
        Calculate the frequency evolution of each of the waveform modes (up to
        ellMax), and of the projected signal (in units of cycles/M) by the 
        user requested method. 
        
        The results are stored to the foft dictionary. This has keys for each 
        [l,m] mode, and an additional key 'signal'. 

        Parameters
        ----------
        method : str, optional
            The method used to calculate the evolving frequency. 
            
            Options are:
                
            - 'phase_derivative'. The frequency is calculated by taking the 
              derivative of the waveform phase. 
            - 'zero_crossings'. The frequency is calculated by counting the 
              zero crossings of the waveform. Within each mode the plus and
              cross frequency evolution is stored separately.
              
            The default is 'phase_derivative'.
        """
        # Dictionary to store the frequency evolution
        self.foft = {}
        
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):
                
                # Load the mode data
                data = self.h[l,m]
                
                if method == 'phase_derivative':
                    
                    # Calculate the phase of the complex data
                    phase = np.unwrap(np.angle(data))
                    
                    # Calculate the derivative using splines
                    phasedot = spline(self.times, phase).derivative()(self.times)
                    
                    # Store to dictionary (dividing by 2*pi to convert to Hz)
                    self.foft[l,m] = np.abs(phasedot)/(2*np.pi)
                
                elif method == 'zero_crossings':
                
                    # Initialize the dictionary
                    self.foft[l,m] = {}
    
                    # Work with the plus and cross components of the data 
                    # separately
                    plus_data = np.real(data)
                    cross_data = -np.imag(data)
                    
                    # Create splines, from which we can find roots (the times 
                    # of the zero-crossings)
                    plus_roots = spline(self.times, plus_data).roots()
                    cross_roots = spline(self.times, cross_data).roots()
                    
                    # The difference between each of these times will be ~T/2, 
                    # so we multiply by 2 to get ~T(t)
                    plus_Toft = 2*np.diff(plus_roots)
                    cross_Toft = 2*np.diff(cross_roots)
                    
                    # The reciprocal gives the frequency evolution
                    plus_foft = 1/plus_Toft
                    cross_foft = 1/cross_Toft
                    
                    # We associate these frequencies with times that are at the 
                    # midpoints of the crossing times
                    plus_foft_times = []
                    cross_foft_times = []
                    
                    for i in range(len(plus_roots)-1):
                        plus_foft_times.append( (plus_roots[i]+plus_roots[i+1])/2 )   
     
                    for i in range(len(cross_roots)-1):
                        cross_foft_times.append( (cross_roots[i]+cross_roots[i+1])/2 )
                    
                    # Store to dictionary
                    self.foft[l,m]['plus'] = np.transpose([plus_foft_times, plus_foft])
                    self.foft[l,m]['cross'] = np.transpose([cross_foft_times, cross_foft])
    
    
    def ringdown_fit(self, t0=0, T=100, Mf=None, chif=None,
                     modes=[(2,2,n) for n in range(8)], mirror_modes=[], 
                     hlm_modes=[(2,2)], t0_method='geq'):
        """
        Perform a least squares fit to the simulation data using a ringdown 
        model.

        Parameters
        ----------
        t0 : float, optional
            The start time of the ringdown model, relative to the chosen zero
            time. The default is 0.
            
        T : float, optional
            The end time of the analysis, relative to t0. The default is 100.
            
        Mf : float, optional
            The remnant black hole mass, which along with chif determines the 
            QNM frequencies. If None, the true remnant mass is used. The 
            default is None.
            
        chif : float, optional
            The magnitude of the remnant black hole spin. If None, the true
            remnant spin is used. The default is None.
            
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
            
        t0_method: str, optional
            A requested ringdown start time will in general lie between times
            on the default time array (the same is true for the end time of
            the analysis). There are different approaches to deal with this, 
            which can be specified here.
            
            Options are:
                
                - 'geq'
                    Take data at times greater than or equal to t0. Note that
                    we still treat the ringdown start time as occuring at t0,
                    so the best fit coefficients are defined with respect to 
                    t0.

                - 'closest'
                    Identify the data point occuring at a time closest to t0, 
                    and take times from there.
                    
                - 'interpolated'
                    Interpolate the data and evaluate on a new array of times,
                    with t0 being the first time. The new time array is 
                    uniformly spaced with separation self.min_dt (the 
                    minimum time between samples in the original data).
                    
            The default is 'geq'.

        Returns
        -------
        best_fit : dict
            A dictionary of useful information related to the fit. Keys 
            include:
                
                - 'residual' : float
                    The residual from the fit.
                - 'mismatch' : float
                    The mismatch between the best-fit waveform and the data.
                - 'C' : array
                    The (shared) best-fit complex amplitudes. There is a 
                    complex amplitude for each ringdown mode.
                - 'weighted_C' : dict
                    The complex amplitudes weighted by the mixing coefficients 
                    and remnant mass. There is a dictionary entry for each hlm
                    mode.
                - 'data' : dict
                    The (masked) data used in the fit.
                - 'model': dict
                    The best-fit model waveform. Keys correspond to the hlm
                    modes.
                - 'model_times' : array
                    The times at which the model is evaluated.
                - 't0' : float
                    The ringdown start time used in the fit.
                - 'modes' : list
                    The regular ringdown modes used in the fit.
                - 'mirror_modes' : list
                    The mirror ringdown modes used in the fit.
                - 'mode_labels' : list
                    Labels for each of the ringdown modes (used for plotting).
                - 'frequencies' : array
                    The values of the complex frequencies for all the ringdown 
                    modes. The order is [modes, mirror_modes].
        """
        # If no mass or spin are given, use the true values:
        if Mf is None:
            Mf = self.Mf
        
        if chif is None:
            chif = self.chif_mag
        
        if t0_method == 'geq':
            
            data_mask = (self.times>=t0) & (self.times<t0+T)
            
            times = self.times[data_mask]
            data = np.concatenate(
                [self.h[lm][data_mask] for lm in hlm_modes])
            
        elif t0_method == 'closest':
            
            start_index = np.argmin((self.times-t0)**2)
            end_index = np.argmin((self.times-t0-T)**2)
            
            times = self.times[start_index:end_index]
            data = np.concatenate(
                [self.h[lm][start_index:end_index] for lm in hlm_modes])
            
        elif t0_method == 'interpolated':
            
            times = np.arange(t0, t0+T, self.min_dt)
            data = np.concatenate(
                [self.h_interp(*lm, times) for lm in hlm_modes])
            
        else:
            print("""Requested t0_method is not valid. Please choose between
                  'geq', 'closest' and 'interpolated'.""")
        
        # Frequencies
        # -----------
        
        # The regular (positive real part) frequencies
        frequencies = np.array(qnm.omega_list(modes, chif, Mf, interp=True))
        
        # The mirror (negative real part) frequencies can be obtained using 
        # symmetry properties 
        mirror_frequencies = -np.conjugate(qnm.omega_list(
            [(l,-m,n) for l,m,n in mirror_modes], chif, Mf))
        
        all_frequencies = np.hstack((frequencies, mirror_frequencies))
        
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
            [hlm_mode+mode for mode in modes] for hlm_mode in hlm_modes]
        
        # Convert each tuple of indices in indices_lists to a mu value
        reg_mu_lists = np.conjugate([
            qnm.mu_list(indices, chif, interp=True) for indices in reg_indices_lists])
        
        # Mirror mixing coefficients
        # --------------------------
            
        # A list of lists for the mixing coefficient indices, see above
        mirror_indices_lists = [
            [(l,-m)+(L,-M,N) for L,M,N in mirror_modes] for l,m in hlm_modes]
        
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
            for i in range(len(hlm_modes))]
        
        # for i, hlm_mode in enumerate(hlm_modes):
        #     mu_lists.append(list(reg_mu_lists[i]) + list(mirror_mu_lists[i]))
            
        # Construct coefficient matrix and solve
        # --------------------------------------
        
        # Construct the coefficient matrix # (Mf/self.M)*
        a = np.concatenate([np.array([
            mu_lists[i][j]*np.exp(-1j*all_frequencies[j]*(times-t0)) 
            for j in range(len(all_frequencies))]).T 
            for i in range(len(hlm_modes))])

        # Solve for the complex amplitudes, C. Also returns the sum of
        # residuals, the rank of a, and singular values of a.
        C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
        
        # Evaluate the model. This needs to be split up into the separate
        # spherical harmonic modes.
        model = np.einsum('ij,j->i', a, C)
        
        # Split up the result into the separate spherical harmonic modes, and
        # store to a dictionary. We put the (masked) data into a dictionary
        # for the mismatch calculation. We also store the "weighted" complex
        # amplitudes to a dictionary.
        model_dict = {}
        data_dict = {}
        
        weighted_C = {}
        
        for i, lm in enumerate(hlm_modes):
            model_dict[lm] = model[i*len(times):(i+1)*len(times)]
            data_dict[lm] = data[i*len(times):(i+1)*len(times)]
            
            weighted_C[lm] = np.array(mu_lists[i])*C # (Mf/self.M)*
        
        # Calculate the (sky averaged) mismatch for the fit
        mm = sky_averaged_mismatch(times, model_dict, data_dict)
        
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
            'frequencies': all_frequencies
            }
        
        # Return the output dictionary
        return best_fit
    
    
    def dynamic_ringdown_fit(self, t0=0, T=100, Moft=None, chioft=None, 
                             modes=[(2,2,n) for n in range(8)], 
                             mirror_modes=[], hlm_modes=[(2,2)]):
        """
        Perform a least squares fit to the simulation data using a dynamic
        ringdown model.

        Parameters
        ----------
        t0 : float, optional
            The start time of the ringdown model, relative to the chosen zero
            time. The default is 0.
            
        T : float, optional
            The end time of the analysis, relative to t0. The default is 100.
            
        Moft : float or array, optional
            The remnant black hole mass, which along with chif determines the 
            QNM frequencies. This can be a float, so that mass doesn't change
            with time, or an array of the same length as self.time. If None, 
            the integrated black hole mass is used (stored as self.Moft). The 
            default is None.
            
        chioft : float or array, optional
            The magnitude of the remnant black hole spin. As with Moft, this 
            can be a float or an array. If None, the integrated black hole 
            spin is used (stored as self.chioft_mag). The default is None.
            
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
        best_fit : dict
            A dictionary of useful information related to the fit. Keys 
            include:
                
                - 'residual' : float
                    The residual from the fit.
                - 'mismatch' : float
                    The mismatch between the best-fit waveform and the data.
                - 'C' : array
                    The (shared) best-fit complex amplitudes. There is a 
                    (time dependant) complex amplitude for each ringdown mode.
                - 'weighted_C' : dict
                    The complex amplitudes weighted by the mixing coefficients 
                    and remnant mass. There is a dictionary entry for each hlm
                    mode.
                - 'data' : dict
                    The (masked) data used in the fit.
                - 'model': dict
                    The best-fit model waveform. Keys correspond to the hlm
                    modes.
                - 'model_times' : array
                    The times at which the model is evaluated.
                - 't0' : float
                    The ringdown start time used in the fit.
                - 'modes' : list
                    The regular ringdown modes used in the fit.
                - 'mirror_modes' : list
                    The mirror ringdown modes used in the fit.
                - 'mode_labels' : list
                    Labels for each of the ringdown modes (used for plotting).
                - 'frequencies' : array
                    The values of the complex frequencies for all the ringdown 
                    modes. The order is [modes, mirror_modes].
        """
        # Mask to cut data of length T from the chosen start time
        data_mask = (self.times>=t0) & (self.times<t0+T)
        times = self.times[data_mask]
        
        # Use the appropriate data
        data = np.concatenate(
            [self.h[lm][data_mask] for lm in hlm_modes])

        # If no mass or spin are given, use the (masked) dynamic values
        if Moft is None:
            Moft = self.Moft[data_mask]
        elif type(Moft) in [list, np.ndarray]:
            Moft = Moft[data_mask]
        
        if chioft is None:
            chioft = self.chioft_mag[data_mask]
        elif type(chioft) in [float, np.float64]:
            chioft = np.full(len(times), chioft)
        else:
            chioft = chioft[data_mask]
        
        # Frequencies
        # -----------
        
        # The regular (positive real part) frequencies
        frequencies = np.array(qnm.omegaoft_list(modes, chioft, Moft))
        
        # The mirror (negative real part) frequencies can be obtained using 
        # symmetry properties 
        mirror_frequencies = -np.conjugate(qnm.omegaoft_list(
            [(l,-m,n) for l,m,n in mirror_modes], chioft, Moft))
        
        if len(mirror_modes) == 0:
            all_frequencies = frequencies.T
        elif len(modes) == 0:
            all_frequencies = mirror_frequencies.T
        else:
            all_frequencies = np.hstack((frequencies.T, mirror_frequencies.T))
        
        # We stack as many frequency arrays on top of each other as we have
        # hlm_modes
        all_frequencies = np.vstack(len(hlm_modes)*[all_frequencies])
            
        # Construct the coefficient matrix for use with NumPy's lstsq 
        # function. We deal with the regular mode and mirror mode mixing
        # coefficients separately.
        
        # Regular mixing coefficients
        # ---------------------------
        
        if len(modes) != 0:
        
            # A list of lists for the mixing coefficient indices. The first 
            # list is associated with the first hlm mode. The second list is 
            # associated with the second hlm mode, and so on.
            # e.g. [ [(2,2,2',2',0'), (2,2,3',2',0')], 
            #        [(3,2,2',2',0'), (3,2,3',2',0')] ]
            reg_indices_lists = [
                [hlm_mode+mode for mode in modes] for hlm_mode in hlm_modes]
            
            # Convert each tuple of indices in indices_lists to an array of mu 
            # values
            reg_mu_lists = np.conjugate([
                qnm.muoft_list(indices, chioft) for indices in reg_indices_lists])
            
            # I = len(hlm_modes)
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
                for i in range(len(hlm_modes))])
            
            # The above reshaping converts the shape into the desired (I*K, J)
            
        if len(mirror_modes) != 0:
        
            # Mirror mixing coefficients
            # --------------------------
                
            # A list of lists for the mixing coefficient indices, see above
            mirror_indices_lists = [
                [(l,-m)+(L,-M,N) for L,M,N in mirror_modes] for l,m in hlm_modes]
            
            # We need to multiply each mu by a factor (-1)**l. Construct these
            # factors from the indices_lists.
            signs = [np.array([
                (-1)**indices[0] for indices in indices_list]) 
                for indices_list in mirror_indices_lists]
            
            # Convert each tuple of indices in indices_lists to a mu value
            mirror_mu_lists = np.array([
                signs[i][:,None]*np.array(qnm.muoft_list(indices, chioft))
                for i, indices in enumerate(mirror_indices_lists)])
            
            # Flatten
            mirror_mu_lists = np.array([
                item for sublist in mirror_mu_lists for item in sublist]).T
            
            # Reshape
            mirror_mu_lists = np.vstack([
                mirror_mu_lists[:,i*len(mirror_modes):(i+1)*len(mirror_modes)] 
                for i in range(len(hlm_modes))])
        
        if len(mirror_modes) == 0:
            mu_lists = reg_mu_lists
        elif len(modes) == 0:
            mu_lists = mirror_mu_lists
        else:
            # Combine the regular and mirror mixing coefficients
            mu_lists = np.hstack((reg_mu_lists, mirror_mu_lists))
        
        # Construct the coefficient matrix
        stacked_times = np.vstack(len(hlm_modes)*[times[:,None]])
        if type(Moft) in [list, np.ndarray]:
            Moft = np.vstack(len(hlm_modes)*[Moft[:,None]])
        a = (Moft/self.M)*mu_lists*np.exp(-1j*all_frequencies*(stacked_times-t0))

        # Solve for the complex amplitudes, C. Also returns the sum of
        # residuals, the rank of a, and singular values of a.
        C, res, rank, s = np.linalg.lstsq(a, data, rcond=None)
        
        # Evaluate the model. This needs to be split up into the separate
        # spherical harmonic modes.
        model = np.einsum('ij,j->i', a, C)
        
        # Evaluate the weighted coefficients (which are now time dependant).
        # These also need to be split up into the separate spherical harmonic
        # modes.
        weighted_C = (Moft/self.M)*mu_lists*C
        
        # Split up the result into the separate spherical harmonic modes, and
        # store to a dictionary. We put the (masked) data into a dictionary
        # for the mismatch calculation. We also store the "weighted" complex
        # amplitudes to a dictionary.
        model_dict = {}
        data_dict = {}
        
        weighted_C_dict = {}
        
        for i, lm in enumerate(hlm_modes):
            model_dict[lm] = model[i*len(times):(i+1)*len(times)]
            data_dict[lm] = data[i*len(times):(i+1)*len(times)]
            
            weighted_C_dict[lm] = weighted_C[i*len(times):(i+1)*len(times)]
        
        # Calculate the (sky averaged) mismatch for the fit
        mm = sky_averaged_mismatch(times, model_dict, data_dict)
        
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
            'frequencies': all_frequencies
            }
        
        # Return the output dictionary
        return best_fit
    
    
    def plot_ringdown(self, hlm_mode=(2,2), xlim=[-50,100], best_fit=None,
                      outfile=None, fig_kw={}):
        """
        Plot the NR data, with an option to plot a best fit model on top.

        Parameters
        ----------
        hlm_mode : tuple, optional
            A (l,m) tuple to specify which spherical harmonic mode to plot. If 
            'signal', the extracted signal is plotted. The default is (2,2).
        
        xlim : tuple, optional
            The x-axis limits. The default is [-50,100].
            
        bestfit : dict, optional
            A bestfit result dictionary from a call of ringdown_fit() or
            dynamic_ringdown_fit(). The 'model' and 'model_times' dictioanary
            entries are accessed to plot the model that matches hlm_mode. If 
            None, no model is plotted. The default is None.
            
        outfile : str, optional
            File name/ path to save the figure. The default is None.
            
        fig_kw : dict, optional
            Additional keyword arguments to pass to plt.subplots() at the 
            figure creation.
        """
        # Use the appropriate data
        if hlm_mode != 'signal':
            data = self.h[hlm_mode]
        else:
            data = self.signal
            
        fig, ax = plt.subplots(figsize=(8,4), **fig_kw)
        
        ax.plot(self.times, np.real(data), 'k-', label=r'$h_+$')
        # ax.plot(self.times, -np.imag(data), 'k--', label=r'$h_\times$')

        if best_fit is not None:
            
            ax.plot(
                best_fit['model_times'], np.real(best_fit['model'][hlm_mode]), 
                'r-', label=r'$h_+$ model', alpha=0.8)
            # ax.plot(
            #     best_fit['model_times'], -np.imag(best_fit['model'][hlm_mode]), 
            #     'r--', label=r'$h_\times$ model', alpha=0.8)

        ax.set_xlim(xlim[0],xlim[1])
        ax.set_xlabel(f'$t\ [M]$ [{self.zero_time_method}]')
        
        if hlm_mode != 'signal':
            ax.set_ylabel(f'$h_{{{hlm_mode[0]}{hlm_mode[1]}}}$')
        else:
            ax.set_ylabel('$h$')

        ax.legend(loc='upper right', frameon=False)
        
        if outfile is not None:
            plt.savefig(outfile)
            plt.close()
            
            
    def plot_ringdown_modes(self, best_fit, hlm_mode=(2,2), xlim=None, 
                            ylim=None, legend=True, outfile=None, fig_kw={}):
        """
        Plot the ringdown waveform from a least squares fit, decomposed into 
        its individual modes.

        Parameters
        ----------
        bestfit : dict
            A bestfit result dictionary from a call of ringdown_fit() or
            dynamic_ringdown_fit(). 
        
        hlm_mode : tuple, optional
            A (l,m) tuple to specify which spherical harmonic mode to plot. 
            The default is (2,2).
            
        xlim : tuple, optional
            The x-axis limits. The default is None.
            
        ylim : tuple, optional
            The y-axis limits. The default is None.
            
        legend : bool, optional
            Toggle the legend on or off. The default is True (legend on).
            
        outfile : str, optional
            File name to save the figure. If None, the figure is not saved. 
            The default is None.
            
        fig_kw : dict, optional
            Additional keyword arguments to pass to plt.subplots() at the 
            figure creation.
        """
        fig, ax = plt.subplots(figsize=(8,4), **fig_kw)
        
        # We sum the modes manually as a check
        mode_sum = np.zeros_like(best_fit['model'][hlm_mode])
        
        for i, mode in enumerate(best_fit['modes'] + best_fit['mirror_modes']):
            
            # The waveform for each mode
            mode_waveform = ringdown(
                best_fit['model_times'], best_fit['t0'], 
                [best_fit['weighted_C'][hlm_mode][i]], 
                [best_fit['frequencies'][i]])
            
            # Add to the overall sum
            mode_sum += mode_waveform
            
            # Use a reduced opacity colour if the colour cycle repeats
            if i > 9:
                alpha = 0.5
            else:
                alpha = 0.7
            
            # Add the mode waveform to the figure. We just plot the real part
            # for clarity.
            ax.plot(
                best_fit['model_times'], np.real(mode_waveform), alpha=alpha)
        
        # The overall sum
        # ax.plot(best_fit['model_times'], np.real(mode_sum), 'k--', alpha=0.7)
        ax.plot(best_fit['model_times'], np.real(mode_sum), 'k--', alpha=1)
        
        if xlim is not None:
            ax.set_xlim(xlim[0],xlim[1])
        ax.set_xlabel(f'$t\ [M]$ [{self.zero_time_method}]')
        
        if ylim is not None:
            ax.set_ylim(ylim[0],ylim[1])
        ax.set_ylabel(f'Re[$h_{{{hlm_mode[0]}{hlm_mode[1]}}}$]')
        
        # Generate the list of labels for the legend
        labels = best_fit['mode_labels'].copy()
        labels.append('Sum')
        
        if legend:
            ax.legend(ax.lines, labels, ncol=3)
        
        if outfile is not None:
            plt.savefig(outfile)
            plt.close()
            
            
    def plot_mode_amplitudes(self, coefficients, labels, log=False, 
                             outfile=None, fig_kw={}):
        """
        Plot the magnitudes of the ringdown modes from a least squares fit.

        Parameters
        ----------
        coefficients : array
            The complex coefficients from a ringdown fit. These are stored as
            bestfit['C'] and bestfit['weighted_C'].
            
        labels : list
            The labels for each coefficient. These are stored as 
            bestfit['mode_labels'] in the ringdown_fit() result dictionary.
        
        log : bool, optional
            If True, use a log scale for the amplitudes. The default is False.
            
        outfile : str, optional
            File name to save the figure. If None, the figure is not saved. 
            The default is None.
            
        fig_kw : dict, optional
            Additional keyword arguments to pass to plt.subplots() at the 
            figure creation.
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
    
    
    def mismatch_t0_array(self, t0_array, T_array=100, Mf=None, chif=None, 
                          modes=[(2,2,n) for n in range(8)], mirror_modes=[], 
                          hlm_modes=[(2,2)], t0_method='geq'):
        """
        Calculate the mismatch for an array of start times.

        Parameters
        ----------
        t0_array : array
            The model start times to compute the mismatch for (relative to the 
            zero time).
            
        T_array : float or array, optional
            The end time of the analysis, relative to t0. If an array, this 
            should be the same length as t0_array. The default is 100.
            
        Mf : float, optional
            The remnant black hole mass, which along with chif determines the 
            QNM frequencies. If None, the true remnant mass is used. The 
            default is None.
            
        chif : float, optional
            The magnitude of the remnant black hole spin. If None, the true
            remnant spin is used. The default is None.
            
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
            
        t0_method: str, optional
            A requested ringdown start time will in general lie between times
            on the default time array (the same is true for the end time of
            the analysis). There are different approaches to deal with this, 
            which can be specified here.
            
            Options are:
                
                - 'geq'
                    Take data at times greater than or equal to t0. Note that
                    we still treat the ringdown start time as occuring at t0,
                    so the best fit coefficients are defined with respect to 
                    t0.

                - 'closest'
                    Identify the data point occuring at a time closest to t0, 
                    and take times from there.
                    
                - 'interpolated'
                    Interpolate the data and evaluate on a new array of times,
                    with t0 being the first time. The new time array is 
                    uniformly spaced with separation self.min_dt (the 
                    minimum time between samples in the original data).
                    
            The default is 'geq'.

        Returns
        -------
        mm_list : array
            The mismatch for each t0 value.
        """
        # List to store the mismatch from each choice of t0
        mm_list = []
        
        if type(T_array) is not np.ndarray:
            T_array = T_array*np.ones(len(t0_array))

        # Now try a range of start times and see what happens to the mismatch
        for t0, T in zip(t0_array, T_array):

            # Run the fit
            best_fit = self.ringdown_fit(
                t0=t0, T=T, Mf=Mf, chif=chif, modes=modes, 
                mirror_modes=mirror_modes, hlm_modes=hlm_modes, 
                t0_method=t0_method)

            # Append the mismatch to the list
            mm_list.append(best_fit['mismatch'])
                        
        return mm_list
    
    
    def dynamic_mismatch_t0_array(self, t0_array, T=100, Moft=None, chioft=None, 
                                  modes=[(2,2,n) for n in range(8)], 
                                  mirror_modes=[], hlm_modes=[(2,2)]):
        """
        Calculate the mismatch for an array of start times, performing the 
        fits with the dynamic_ringdown_fit() function.

        Parameters
        ----------
        t0_array : array
            The model start times to compute the mismatch for (relative to the 
            zero time).
            
        T : float, optional
            The end time of the analysis, relative to t0. The default is 100.
            
        Moft : float or array, optional
            The remnant black hole mass, which along with chif determines the 
            QNM frequencies. This can be a float, so that mass doesn't change
            with time, or an array of the same length as self.time. If None, 
            the integrated black hole mass is used (stored as self.Moft). The 
            default is None.
            
        chioft : float or array, optional
            The magnitude of the remnant black hole spin. As with Moft, this 
            can be a float or an array. If None, the integrated black hole 
            spin is used (stored as self.chioft_mag). The default is None.
            
        modes : list, optional
            A list of (l,m,n) tuples (where l is the angular number of the 
            mode, m is the azimuthal number, and n is the overtone number) 
            specifying which modes to use in the model. The default is 
            [(2,2,0),(3,2,0)].
            
        mirror_modes : list, optional
            A list of (l,m,n) tuples (as in modes), specifying which 'mirror
            modes' to use in the model. The default is [] (no mirror modes are
            included).
            
        hlm_modes : list, optional
            A list of (l,m) tuples to specify which spherical harmonic hlm 
            modes the analysis should be performed on. The default is [(2,2)].

        Returns
        -------
        mm_list : array
            The mismatch for each t0 value.
        """
        # List to store the mismatch from each choice of t0
        mm_list = []

        # Now try a range of start times and see what happens to the mismatch
        for t0 in t0_array:

            # Run the fit
            best_fit = self.dynamic_ringdown_fit(
                t0=t0, T=T, Moft=Moft, chioft=chioft, modes=modes, 
                mirror_modes=mirror_modes, hlm_modes=hlm_modes)

            # Append the mismatch to the list
            mm_list.append(best_fit['mismatch'])
                        
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
        self.Mf_array = np.linspace(Mf_minmax[0], Mf_minmax[1], res)
        self.chif_array = np.linspace(chif_minmax[0], chif_minmax[1], res)

        # List to store the mismatch from each choice of M and chi
        mm_list = []

        # Cycle through each combination of mass and spin, calculating the
        # mismatch for each. Use a single loop for the progress bar.
        for i in tqdm(range(len(self.Mf_array)*len(self.chif_array))):

            # Get the mass and spin values from the value of i
            Mf = self.Mf_array[int(i/len(self.Mf_array))]
            chif = self.chif_array[i%len(self.chif_array)]
            
            # Run the fit
            bestfit = self.ringdown_fit(
                t0=t0, T=T, Mf=Mf, chif=chif, modes=modes, 
                mirror_modes=mirror_modes, hlm_modes=hlm_modes)

            # Append the mismatch to the list
            mm_list.append(bestfit['mismatch'])

        # Convert the list of mismatches to a grid
        self.mm_grid = np.reshape(
            np.array(mm_list), (len(self.Mf_array), len(self.chif_array)))
        
        return self.mm_grid
    
    
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