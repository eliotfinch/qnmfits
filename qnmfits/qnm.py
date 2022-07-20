import numpy as np
import qnm as qnm_loader

from scipy.interpolate import interp1d


class qnm:
    """
    Class for loading quasinormal mode (QNM) frequencies and spherical-
    spheroidal mixing coefficients. This makes use of the qnm package,
    https://arxiv.org/abs/1908.10377
    """
    
    def __init__(self):
        """
        Initialise the class.
        """
        
        # Dictionary to store the qnm functions
        self.qnm_funcs = {}
        
        # Dictionary to store interpolated qnm functions for quicker 
        # evaluation
        self.interpolated_qnm_funcs = {}
        
    def interpolate(self, l, m, n):
        
        qnm_func = self.qnm_funcs[l,m,n]
        
        # Extract relevant quantities
        spins = qnm_func.a
        real_omega = np.real(qnm_func.omega)
        imag_omega = np.imag(qnm_func.omega)
        all_real_mu = np.real(qnm_func.C)
        all_imag_mu = np.imag(qnm_func.C)

        # Interpolate omegas
        real_omega_interp = interp1d(
            spins, real_omega, kind='cubic', bounds_error=False, 
            fill_value=(real_omega[0],real_omega[-1]))
        
        imag_omega_interp = interp1d(
            spins, imag_omega, kind='cubic', bounds_error=False, 
            fill_value=(imag_omega[0],imag_omega[-1]))
        
        # Interpolate mus
        mu_interp = []
        
        for real_mu, imag_mu in zip(all_real_mu.T, all_imag_mu.T):
            
            real_mu_interp = interp1d(
                    spins, real_mu, kind='cubic', bounds_error=False, 
                    fill_value=(real_mu[0],real_mu[-1]))
                
            imag_mu_interp = interp1d(
                spins, imag_mu, kind='cubic', bounds_error=False, 
                fill_value=(imag_mu[0],imag_mu[-1]))
            
            mu_interp.append((real_mu_interp, imag_mu_interp))

        # Add these interpolated functions to the frequency_funcs dictionary
        self.interpolated_qnm_funcs[l,m,n] = [
            (real_omega_interp, imag_omega_interp), mu_interp]
        
    def omega(self, l, m, n, chif, Mf=1, interp=False):
        """
        Return a complex frequency, :math:`\omega_{\ell m n}(M_f, \chi_f)`,
        for a particular mass, spin, and mode.
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float, optional
            The mass of the final black hole. This is the factor which the QNM
            frequencies are divided through by, and so determines the units of 
            the returned quantity. 
            
            If Mf is in units of seconds, then the returned frequency has units 
            :math:`\mathrm{s}^{-1}`. 
            
            When working with SXS simulations and GW surrogates, we work in 
            units scaled by the total mass of the binary system, M. In this 
            case, providing the dimensionless Mf value (the final mass scaled 
            by the total binary mass) will ensure the QNM frequencies are in 
            the correct units (scaled by the total binary mass). This is 
            because the frequencies loaded from file are scaled by the remnant 
            black hole mass (Mf*omega). So, by dividing by the remnant black 
            hole mass scaled by the total binary mass (Mf/M), we are left with
            Mf*omega/(Mf/M) = M*omega.
            
            The default is 1, in which case the frequencies are returned in
            units of the remnant black hole mass.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is False.
            
        Returns
        -------
        complex
            The complex QNM frequency.
        """
        # Test if the qnm function has been loaded for the requested mode
        if (l,m,n) not in self.qnm_funcs:
            self.qnm_funcs[l,m,n] = qnm_loader.modes_cache(-2, l, m, n)
        
        # We only cache values if not interpolating
        store = False if interp else True

        # Access the relevant functions from the qnm_funcs dictionary, and 
        # evaluate at the requested spin. Storing speeds up future evaluations.
        omega, A, mu = self.qnm_funcs[l,m,n](
            chif, store=store, interp_only=interp)
        
        # Return the scaled complex frequency
        return omega/Mf
    
    
    def omega_list(self, modes, chif, Mf=1, interp=False):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin)
        
        Parameters
        ----------            
        modes : list
            A list of (l,m,n) tuples (where l is the angular number of the
            mode, m is the azimuthal number, and n is the overtone number)
            specifying which modes to load frequencies for.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float, optional
            The mass of the final black hole. See the qnm.omega() docstring 
            for details on units. The default is 1.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is False.
            
        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the qnm function and append the result to the
        # list
        return [self.omega(l, m, n, chif, Mf, interp) for l, m, n in modes]
    
    
    def omegaoft(self, l, m, n, chioft, Moft=1, interp=True):
        """
        Return an array of complex frequencies corresponding to an array of
        spin and mass values. This is designed to be used with time
        dependant spin and mass values, to get a time dependant frequency
        (hence the name).
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        chioft : array
            The dimensionless spin magnitude of the black hole.
            
        Moft : array, optional
            The time dependant mass of the black hole. See the qnm.omega() 
            docstring for details on units. This can either be a float, which
            then divides through the whole omega array, or an array of the
            same length as chioft. The default is 1, in which case
            no scaling is performed.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is True.
            
        Returns
        -------
        array
            The complex QNM frequency array, with the same length as chioft.
        """
        # Test if the qnm function has been loaded for the requested mode
        if (l,m,n) not in self.qnm_funcs:
            self.qnm_funcs[l,m,n] = qnm_loader.modes_cache(-2, l, m, n)
            
        # Test if the interpolated qnm function has been created
        if (l,m,n) not in self.interpolated_qnm_funcs:
            self.interpolate(l,m,n)
            
        if interp:
            
            # We create our own interpolant so that we can evaulate the 
            # frequencies for all spins simultaneously
            omega_interp = self.interpolated_qnm_funcs[l,m,n][0]
            omegaoft = omega_interp[0](chioft) + 1j*omega_interp[1](chioft)
            
            return omegaoft/Moft
        
        else:
            # List to store the frequencies corresponding to each mass and spin
            omegaoft = []
            
            for chi in chioft:
                
                # Access the relevant functions from the qnm_funcs dictionary, and 
                # evaluate at the requested spin. Storing speeds up future 
                # evaluations.
                omega, A, mu = self.qnm_funcs[l,m,n](chi, store=True)
                
                omegaoft.append(omega)
            
            # Return the scaled complex frequency
            return np.array(omegaoft)/Moft
    
    def omegaoft_list(self, modes, chioft, Moft=1, interp=True):
        """
        Return a list of arrays. Each array in the list contains the time 
        dependant complex frequency (determined by the spin and mass arrays).
        An array is constructed for each mode in the modes list.
        
        Parameters
        ----------            
        modes : list
            A list of (l,m,n) tuples (where l is the angular number of the
            mode, m is the azimuthal number, and n is the overtone number)
            specifying which modes to load frequencies for.
            
        chioft : array
            The dimensionless spin magnitude of the black hole.
            
        Moft : array, optional
            The time dependant mass of the black hole. See the qnm.omega() 
            docstring for details on units. This can either be a float, which
            then divides through the whole omega array, or an array of the
            same length as chioft. The default is 1, in which case
            no scaling is performed.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested 
            frequency. This is faster than calculating the exact value. The
            default is True.
            
        Returns
        -------
        list
            The list of complex QNM frequencies, where each element in the 
            list is an array of length chioft.
        """
        # List to store the frequencies
        omegas = []

        # For each mode, call the qnm function and append the result to the
        # list
        for l, m, n in modes:
            omegas.append(self.omegaoft(l, m, n, chioft, Moft, interp))

        return omegas
    
    def mu(self, l, m, lp, mp, nprime, chif, interp=False):
        """
        Return a spherical-spheroidal mixing coefficient, 
        :math:`\mu_{\ell m \ell' m' n'}(\chi_f)`, for a particular spin and 
        mode combination. The indices (l,m) refer to the spherical harmonic. 
        The indices (l',m',n') refer to the spheroidal harmonic. We use the
        definition of the coefficients in https://arxiv.org/abs/1408.1860.

        Parameters
        ----------
        l : int
            The angular number of the spherical harmonic mode.
            
        m : int
            The azimuthal number of the spherical harmonic mode.
            
        lp : int
            The angular number of the spheroidal harmonic mode.
            
        mp : int
            The azimuthal number of the spheroidal harmonic mode.
            
        nprime : int
            The overtone number of the spheroidal harmonic mode.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested mixing 
            coefficient. This is faster than calculating the exact value. The
            default is False.

        Returns
        -------
        complex
            The spherical-spheroidal mixing coefficient.

        """
        # Test if the qnm function has been loaded for the requested mode
        if (lp,mp,nprime) not in self.qnm_funcs:
            self.qnm_funcs[lp,mp,nprime] = qnm_loader.modes_cache(
                -2, lp, mp, nprime)
            
        # We only cache values if not interpolating
        store = False if interp else True

        # Access the relevant functions from the qnm_funcs dictionary, and 
        # evaluate at the requested spin. Storing speeds up future evaluations.
        omega, A, mu = self.qnm_funcs[lp,mp,nprime](
            chif, store=store, interp_only=interp)
        
        if mp != m:
            # There is no overlap between different values of m
            return 0
        else:
            # mu is an array of all mixing coefficients with the given 
            # (l',m',n'), so we need to index it to get the requested l. We
            # conjugate to match the definition of mu in the above reference.
            if abs(m) > 2:
                index = l - abs(m)
            else:
                index = l - 2
            return np.conjugate(mu[index])
        
    def mu_list(self, indices, chif, interp=False):
        """
        Return a list of mixing coefficients, for all requested indices. See
        the qnm.mu() docstring for more details.
        
        Parameters
        ----------
        indices : list
            A list of (l,m,l',m',n') tuples specifying which mixing 
            coefficients to return.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested mixing 
            coefficient. This is faster than calculating the exact value. The
            default is False.

        Returns
        -------
        mus : list
            The list of spherical-spheroidal mixing coefficients.
        """
        # List to store the mixing coeffs
        mus = []
        
        # For each mode, call the qnm function and append the result to the 
        # list
        for l, m, lp, mp, nprime in indices:
            mus.append(self.mu(l, m, lp, mp, nprime, chif, interp=interp))
        
        return mus
    
    def muoft(self, l, m, lp, mp, nprime, chioft, interp=True):
        """
        Return an array of spherical-spheroidal mixing coefficients, for a 
        array of spins spin and a particular mode combination. See the
        qnm.mu() docstring for more details.

        Parameters
        ----------
        l : int
            The angular number of the spherical harmonic mode.
            
        m : int
            The azimuthal number of the spherical harmonic mode.
            
        lp : int
            The angular number of the spheroidal harmonic mode.
            
        mp : int
            The azimuthal number of the spheroidal harmonic mode.
            
        nprime : int
            The overtone number of the spheroidal harmonic mode.
            
        chioft : array
            The dimensionless spin magnitude of the black hole.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested mixing 
            coefficient. This is faster than calculating the exact value. The
            default is True.

        Returns
        -------
        array
            The spherical-spheroidal mixing coefficients for each value of the
            spin in chioft.

        """
        # Test if the qnm function has been loaded for the requested mode
        if (lp,mp,nprime) not in self.qnm_funcs:
            self.qnm_funcs[lp,mp,nprime] = qnm_loader.modes_cache(
                -2, lp, mp, nprime)
            
        # Test if the interpolated qnm function has been created
        if (lp,mp,nprime) not in self.interpolated_qnm_funcs:
            self.interpolate(lp,mp,nprime)
            
        if interp:
            
            if mp != m:
                
                return np.zeros(len(chioft))
            
            else:
            
                mu_interp = self.interpolated_qnm_funcs[lp,mp,nprime][1][l-2]
                muoft = mu_interp[0](chioft) + 1j*mu_interp[1](chioft)
                
                return np.conjugate(muoft)
            
        else:
            # List to store the mixing coefficients corresponding to each value of
            # the spin
            muoft = []
            
            for chi in chioft:
    
                # Access the relevant functions from the qnm_funcs dictionary, and 
                # evaluate at the requested spin. Storing speeds up future 
                # evaluations.
                omega, A, mu = self.qnm_funcs[lp,mp,nprime](chi, store=True)
            
                if mp != m:
                    # There is no overlap between different values of m
                    muoft.append(0)
                else:
                    # mu is an array of all mixing coefficients with the given 
                    # (l',m',n'), so we need to index it to get the requested l. 
                    # We conjugate to match the definition of mu in the above 
                    # reference.
                    muoft.append(np.conjugate(mu[l-2]))
                    
            return np.array(muoft)
        
    def muoft_list(self, indices, chioft, interp=True):
        """
        Return a list of arrays. Each array in the list conatins the mixing
        coefficients for the array of spin values provided. An array is 
        constructed for each set of provided indices.
        
        Parameters
        ----------
        indices : list
            A list of (l,m,l',m',n') tuples specifying which mixing 
            coefficients to return.
            
        chioft : array
            The dimensionless spin magnitude of the black hole.
            
        interp : bool, optional
            If True, use a simple interpolation to find the requested mixing 
            coefficient. This is faster than calculating the exact value. The
            default is True.

        Returns
        -------
        mus : list
            The list of spherical-spheroidal mixing coefficients, where each
            element of the list is an array of length chioft.
        """
        # List to store the mixing coeffs
        mus = []
        
        # For each mode, call the qnm function and append the result to the 
        # list
        for l, m, lp, mp, nprime in indices:
            mus.append(self.muoft(l, m, lp, mp, nprime, chioft, interp))
        
        return mus
    
    
class qnm_geo:
    """
    Class for calculating the QNM frequencies from geometrical properties, see
    https://arxiv.org/abs/1207.4253
    """
    
    def __init__(self):
        """Initialise the class."""
        
    def omega(self, l, m, n, Omega_theta, Omega_prec, gamma_L):
        """
        Return a complex frequency, :math:`\omega_{\ell m n}(\Omega_\\theta, 
        \Omega_{\mathrm{prec}}, \gamma_L)`.
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        Omega_theta : float
            The orbital frequency of the photon orbit in the polar direction.
            
        Omega_prec : float
            The precession frequency of the orbital plane.
            
        gamma_L : float
            The Lyapunov exponent of the orbit.
            
        Returns
        -------
        complex
            The complex QNM frequency.
        """
        omega_r = (l + 0.5)*Omega_theta + m*Omega_prec
        omega_i = -(n + 0.5)*gamma_L
        
        return omega_r + 1j*omega_i
    
    
    def omega_list(self, modes, Omega_theta, Omega_prec, gamma_L):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin)
        
        Parameters
        ----------            
        modes : list
            A list of (l,m,n) tuples (where l is the angular number of the
            mode, m is the azimuthal number, and n is the overtone number)
            specifying which modes to load frequencies for.
            
        Omega_theta : float
            The orbital frequency of the photon orbit in the polar direction.
            
        Omega_prec : float
            The precession frequency of the orbital plane.
            
        gamma_L : float
            The Lyapunov exponent of the orbit.
            
        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the omega function and append the result to the
        # list
        return [self.omega(l, m, n, Omega_theta, Omega_prec, gamma_L) 
                for l, m, n in modes]
    
    def omega_mod(self, l, m, n, Omega_theta, Omega_prec, gamma_L):
        """
        Return a complex frequency, :math:`\omega_{\ell m n}(\Omega_\\theta, 
        \Omega_{\mathrm{prec}}, \gamma_L)`, with the correction introduced in
        https://arxiv.org/abs/2104.07594
        
        Parameters
        ----------
        l : int
            The angular number of the mode.
            
        m : int
            The azimuthal number of the mode.
            
        n : int
            The overtone number of the mode.
            
        Omega_theta : float
            The orbital frequency of the photon orbit in the polar direction.
            
        Omega_prec : float
            The precession frequency of the orbital plane.
            
        gamma_L : float
            The Lyapunov exponent of the orbit.
            
        Returns
        -------
        complex
            The complex QNM frequency.
        """
        omega_r = l*(Omega_theta + (m/(l+0.5))*Omega_prec)
        omega_i = -(l+l**2+l**3)/(1+l+l**2+l**3)*(n + 0.5)*gamma_L
        
        return omega_r + 1j*omega_i
    
    
    def omega_mod_list(self, modes, Omega_theta, Omega_prec, gamma_L):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin)
        
        Parameters
        ----------            
        modes : list
            A list of (l,m,n) tuples (where l is the angular number of the
            mode, m is the azimuthal number, and n is the overtone number)
            specifying which modes to load frequencies for.
            
        Omega_theta : float
            The orbital frequency of the photon orbit in the polar direction.
            
        Omega_prec : float
            The precession frequency of the orbital plane.
            
        gamma_L : float
            The Lyapunov exponent of the orbit.
            
        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the omega function and append the result to the
        # list
        return [self.omega_mod(l, m, n, Omega_theta, Omega_prec, gamma_L) 
                for l, m, n in modes]
    
    
class Elmn:

    def __init__(self):
        self.data = np.array([
            [0      ,  0.0743 ,  0.210 ,  0.0850 ,  0.340 ,  0.0983 ,  0.488 ,  0.115 ,  0.662  ,  0.136 ,  0.879 ,  0.164 ,  1.17 ,  0.194 ,  1.62 ,  0.148 ,  2.71 ],
            [1      ,  0.230 ,  -2.14 ,  0.280 ,  -1.99 ,  0.346 ,  -1.81 ,  0.435 ,  -1.59 ,  0.557 ,  -1.31 ,  0.725 ,  -0.917 ,  0.927 ,  -0.292 ,  0.716 ,  1.15],
            [2      ,  0.417 ,  1.61 ,  0.537 ,  1.78 ,  0.703 ,  2.00 ,  0.942 ,  2.27 ,  1.29 ,  2.64 ,  1.80 ,  -3.10 ,  2.39 ,  -2.22 ,  1.81 ,  -0.285],
            [3      ,  0.585 ,  -1.08 ,  0.788 ,  -0.885 ,  1.08 ,  -0.635 ,  1.54 ,  -0.307 ,  2.28 ,  0.155 ,  3.50 ,  0.885 ,  4.63 ,  2.12 ,  3.25 ,  -1.62],
            [4      ,  0.725 ,  2.32 ,  0.996 ,  2.54 ,  1.40 ,  2.82 ,  2.05 ,  -3.09 ,  3.26 ,  -2.56 ,  6.01 ,  -1.68 ,  8.40 ,  0.250 ,  4.68 ,  -2.88],
            [5      ,  0.832 ,  -0.696 ,  1.12 ,  -0.462 ,  1.53 ,  -0.152 ,  2.20 ,  0.258 ,  3.44 ,  0.825 ,  6.75 ,  1.70 ,  21.6 ,  2.60 ,  0.505 ,  0.747],
            [6      ,  0.881 ,  2.46 ,  1.12 ,  2.73 ,  1.45 ,  3.10 ,  1.94 ,  -2.69 ,  2.77 ,  -2.04 ,  4.72 ,  -1.13 ,  20.8 ,  -0.605 ,  5.76 ,  2.22],
            [7      ,  0.854 ,  -0.702 ,  1.02 ,  -0.360 ,  1.25 ,  0.0932 ,  1.55 ,  0.689 ,  2.02 ,  1.49 ,  2.93 ,  2.62 ,  6.31 ,  -2.01 ,  6.30 ,  1.10],
            [8      ,  0.781 ,  2.43 ,  0.898 ,  2.85 ,  1.04 ,  -2.87 ,  1.23 ,  -2.16 ,  1.48 ,  -1.20 ,  1.92 ,  0.170 ,  3.36 ,  2.34 ,  6.28 ,  0.0509],
            [9      ,  0.700 ,  -0.678 ,  0.783 ,  -0.179 ,  0.877 ,  0.454 ,  0.973 ,  1.28 ,  1.09 ,  2.40 ,  1.28 ,  -2.26 ,  1.96 ,  0.362 ,  5.80 ,  -0.94 ],
            [10      ,  0.628 ,  2.51 ,  0.686 ,  3.07 ,  0.737 ,  -2.49 ,  0.774 ,  -1.55 ,  0.802 ,  -0.278 ,  0.860 ,  1.58 ,  1.17 ,  -1.63 ,  5.03 ,  -1.87],
            [11      ,  0.569 ,  -0.567 ,  0.605 ,  0.0565 ,  0.623 ,  0.847 ,  0.616 ,  1.89 ,  0.591 ,  -2.96 ,  0.576 ,  -0.852 ,  0.701 ,  2.66 ,  4.11 ,  -2.76],
            [12      ,  0.519 ,  2.64 ,  0.537 ,  -2.96 ,  0.528 ,  -2.10 ,  0.491 ,  -0.950 ,  0.435 ,  0.640 ,  0.384 ,  2.99 ,  0.419 ,  0.657 ,  3.18 ,  2.68],
            [13      ,  0.477 ,  -0.429,  0.478 ,  0.297 ,  0.448 ,  1.23 ,  0.390 ,  2.49 ,  0.318 ,  -2.04 ,  0.254 ,  0.550 ,  0.249 ,  -1.34 ,  2.33 ,  1.87],
            [14      ,  0.440 ,  2.78 ,  0.427 ,  -2.72 ,  0.381 ,  -1.71 ,  0.310 ,  -0.356 ,  0.231 ,  1.54 ,  0.167 ,  -1.89 ,  0.147 ,  2.94 ,  1.62 ,  1.06 ],
            [15      ,  0.408 ,  -0.292 ,  0.382 ,  0.533 ,  0.323 ,  1.61 ,  0.245 ,  3.07 ,  0.168 ,  -1.14 ,  0.109 ,  1.94 ,  0.0869 ,  0.936 ,  1.09 ,  0.268 ],
            [16      ,  0.377 ,  2.91 ,  0.342 ,  -2.49 ,  0.274 ,  -1.34 ,  0.193 ,  0.228 ,  0.121 ,  2.44 ,  0.0714 ,  -0.500 ,  0.0508 ,  -1.06 ,  0.710 ,  -0.534],
            [17      ,  0.350 ,  -0.159 ,  0.307 ,  0.763 ,  0.233 ,  1.98 ,  0.152 ,  -2.62 ,  0.0873 ,  -0.249 ,  0.0461 ,  -2.95 ,  0.0295 ,  -3.07 ,  0.453 ,  -1.34],
            [18      ,  0.327 ,  3.04 ,  0.273 ,  -2.26 ,  0.195 ,  -0.970 ,  0.119 ,  0.805 ,  0.0624 ,  -2.94 ,  0.0296 ,  0.883 ,  0.0171 ,  1.21 ,  0.285 ,  -2.16 ],
            [19      ,  0.307 ,  -0.0196 ,  0.245 ,  0.989 ,  0.167 ,  2.34 ,  0.0934 ,  -2.04 ,  0.0443 ,  0.635 ,  0.0189 ,  -1.57 ,  0.00983 ,  -0.792 ,  0.178 ,  -2.98 ],
            [20      ,  0.284 ,  -3.11 ,  0.219 ,  -2.04 ,  0.140 ,  -0.593 ,  0.0726 ,  1.37 ,  0.0313 ,  -2.07 ,  0.0120 ,  2.24 ,  0.00563 ,  -2.80 ,  0.110 ,  2.48]
        ])

        self.setup_interps()


    def setup_interps(self):

        j_vals = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

        amps = self.data[:,1::2]
        phis = self.data[:,2::2]

        self.amp_interps = [ interp1d(j_vals, amps[n], kind='cubic') for n in range(21)]
        self.phi_interps = [ interp1d(j_vals, phis[n], kind='cubic') for n in range(21)]

    def Elmn(self, j, l=2, m=2, n=0):

        return self.amp_interps[n](j)*np.exp((1j)*self.phi_interps[n](j))
    
    def Elmn_list(self, indices, chif):
        
        Es = []
        
        # For each mode, call the qnm function and append the result to the 
        # list
        for l, m, n in indices:
            Es.append(self.Elmn(chif, l, m, n))
            
        return Es