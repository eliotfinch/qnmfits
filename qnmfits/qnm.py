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
        
    def omega(self, l, m, n, sign, chif, Mf=1, interp=False):
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
            
        sign : int
            An integer with value +1 or -1, to indicate the sign of the real
            part of the frequency. This way any regular (+1) or mirror (-1)
            mode can be requested. Alternatively, this can be thought of as
            prograde (sign = sgn(m)) or retrograde (sign = -sgn(m)) modes.
            
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
        # Load the correct qnm based on the type we want
        m *= sign
        
        # Test if the qnm function has been loaded for the requested mode
        if (l,m,n) not in self.qnm_funcs:
            self.qnm_funcs[l,m,n] = qnm_loader.modes_cache(-2, l, m, n)
        
        # We only cache values if not interpolating
        store = False if interp else True

        # Access the relevant functions from the qnm_funcs dictionary, and 
        # evaluate at the requested spin. Storing speeds up future evaluations.
        omega, A, mu = self.qnm_funcs[l,m,n](
            chif, store=store, interp_only=interp)
        
        # Use symmetry properties to get the mirror mode, if requested
        if sign == -1:
            omega = -np.conjugate(omega)
        
        # Return the scaled complex frequency
        return omega/Mf
    
    def omega_list(self, modes, chif, Mf=1, interp=False):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin)
        
        Parameters
        ----------            
        modes : array_like
            A sequence of (l,m,n,sign) tuples to specify which QNMs to load 
            frequencies for. For nonlinear modes, the tuple has the form 
            (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
            
        chif : float
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float, optional
            The mass of the final black hole. See the qnm.omega docstring for
            details on units. The default is 1.
            
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
        
        # Code for linear QNMs:
        # return [self.omega(l, m, n, sign, chif, Mf, interp) for l, m, n, sign in modes]
        
        # Code for nonlinear QNMs:
        return [
            sum([self.omega(l, m, n, sign, chif, Mf, interp) 
                 for l, m, n, sign in [mode[i:i+4] for i in range(0, len(mode), 4)]
                 ]) 
            for mode in modes
            ]
    
        # Writen out, the above is doing the following:
            
        # return_list = []
        # for mode in modes:
        #     sum_list = []
        #     for i in range(0, len(mode), 4):
        #         l, m, n, sign = mode[i:i+4]
        #         sum_list.append(self.omega(l, m, n, sign, chif, Mf, interp))
        #     return_list.append(sum(sum_list))
        # return return_list
    
    def omegaoft(self, l, m, n, sign, chioft, Moft=1, interp=True):
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
            
        sign : int
            An integer with value +1 or -1, to indicate the sign of the real
            part of the frequency. This way any regular (+1) or mirror (-1)
            mode can be requested. Alternatively, this can be thought of as
            prograde (sign = sgn(m)) or retrograde (sign = -sgn(m)) modes.
            
        chioft : array_like
            The dimensionless spin magnitude of the black hole.
            
        Moft : array_like, optional
            The time dependant mass of the black hole. See the qnm.omega
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
        ndarray
            The complex QNM frequency array, with the same length as chioft.
        """
        # Load the correct qnm based on the type we want
        m *= sign
        
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
            
            # Use symmetry properties to get the mirror mode, if requested
            if sign == -1:
                omegaoft = -np.conjugate(omegaoft)
            
            return omegaoft/Moft
        
        else:
            # List to store the frequencies corresponding to each mass and spin
            omegaoft = []
            
            for chi in chioft:
                
                # Access the relevant functions from the qnm_funcs dictionary, 
                # and evaluate at the requested spin. Storing speeds up future 
                # evaluations.
                omega, A, mu = self.qnm_funcs[l,m,n](chi, store=True)
                
                omegaoft.append(omega)
                
            # Use symmetry properties to get the mirror mode, if requested
            if sign == -1:
                omegaoft = -np.conjugate(omegaoft)
            
            # Return the scaled complex frequency
            return np.array(omegaoft)/Moft
    
    def omegaoft_list(self, modes, chioft, Moft=1, interp=True):
        """
        Return a list of arrays. Each array in the list contains the time 
        dependant complex frequency (determined by the spin and mass arrays).
        An array is constructed for each mode in the modes list.
        
        Parameters
        ----------            
        modes : array_like
            A sequence of (l,m,n,sign) tuples to specify which QNMs to load 
            frequencies for. For nonlinear modes, the tuple has the form 
            (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
            
        chioft : array_like
            The dimensionless spin magnitude of the black hole.
            
        Moft : array_like, optional
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
        # Code for linear QNMs:
        # omegas = []
        # for l, m, n, sign in modes:
        #     omegas.append(self.omegaoft(l, m, n, sign, chioft, Moft, interp))
        # return omegas
            
        # Code for nonlinear QNMs:
        return [
            sum([self.omegaoft(l, m, n, sign, chioft, Moft, interp) 
                 for l, m, n, sign in [mode[i:i+4] for i in range(0, len(mode), 4)]
                 ]) 
            for mode in modes
            ]
    
    def mu(self, l, m, lp, mp, nprime, sign, chif, interp=False):
        """
        Return a spherical-spheroidal mixing coefficient, 
        :math:`\mu_{\ell m \ell' m' n'}(\chi_f)`, for a particular spin and 
        mode combination. The indices (l,m) refer to the spherical harmonic. 
        The indices (l',m',n') refer to the spheroidal harmonic.

        Parameters
        ----------
        l : int
            The angular number of the spherical-harmonic mode.
            
        m : int
            The azimuthal number of the spherical-harmonic mode.
            
        lp : int
            The angular number of the spheroidal-harmonic mode.
            
        mp : int
            The azimuthal number of the spheroidal-harmonic mode.
            
        nprime : int
            The overtone number of the spheroidal-harmonic mode.
            
        sign : int
            
            
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
        # There is no overlap between different values of m
        if mp != m:
            return 0
        
        # Load the correct qnm based on the type we want
        m *= sign
        mp *= sign
        
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

        # mu is an array of all mixing coefficients with the given (l',m',n'), 
        # so we need to index it to get the requested l
        if abs(m) > 2:
            index = l - abs(m)
        else:
            index = l - 2
            
        mu = mu[index]
        
        # Use symmetry properties to get the mirror mixing coefficient, if 
        # requested
        if sign == -1:
            mu = (-1)**(l+lp)*np.conjugate(mu)
            
        return mu
        
    def mu_list(self, indices, chif, interp=False):
        """
        Return a list of mixing coefficients, for all requested indices. See
        the qnm.mu() docstring for more details.
        
        Parameters
        ----------
        indices : array_like
            A sequence of (l,m,l',m',n') tuples specifying which mixing 
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
        for l, m, lp, mp, nprime, sign in indices:
            mus.append(self.mu(l, m, lp, mp, nprime, sign, chif, interp=interp))
        
        return mus
    
    def muoft(self, l, m, lp, mp, nprime, sign, chioft, interp=True):
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
            
        chioft : array_like
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
        # There is no overlap between different values of m
        if mp != m:
            return np.zeros(len(chioft))
            
        # Load the correct qnm based on the type we want
        m *= sign
        mp *= sign
        
        # Test if the qnm function has been loaded for the requested mode
        if (lp,mp,nprime) not in self.qnm_funcs:
            self.qnm_funcs[lp,mp,nprime] = qnm_loader.modes_cache(
                -2, lp, mp, nprime)
            
        # Test if the interpolated qnm function has been created
        if (lp,mp,nprime) not in self.interpolated_qnm_funcs:
            self.interpolate(lp,mp,nprime)
            
        # Our functions return all mixing coefficients with the given 
        # (l',m',n'), so we need to index it to get the requested l
        if abs(m) > 2:
            index = l - abs(m)
        else:
            index = l - 2
            
        if interp:
            mu_interp = self.interpolated_qnm_funcs[lp,mp,nprime][1][index]
            muoft = mu_interp[0](chioft) + 1j*mu_interp[1](chioft)
            
        else:
            # List to store the mixing coefficients corresponding to each 
            # value of the spin
            muoft = []
            
            for chi in chioft:
                # Access the relevant functions from the qnm_funcs dictionary, 
                # and evaluate at the requested spin. Storing speeds up future 
                # evaluations.
                omega, A, mu = self.qnm_funcs[lp,mp,nprime](chi, store=True)
                muoft.append(mu[index])
                
        # Use symmetry properties to get the mirror mixing coefficient, if 
        # requested
        if sign == -1:
            muoft = (-1)**(l+lp)*np.conjugate(muoft)
                    
        return np.array(muoft)
        
    def muoft_list(self, indices, chioft, interp=True):
        """
        Return a list of arrays. Each array in the list conatins the mixing
        coefficients for the array of spin values provided. An array is 
        constructed for each set of provided indices.
        
        Parameters
        ----------
        indices : array_like
            A sequence of (l,m,l',m',n') tuples specifying which mixing 
            coefficients to return.
            
        chioft : array_like
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
        for l, m, lp, mp, nprime, sign in indices:
            mus.append(self.muoft(l, m, lp, mp, nprime, sign, chioft, interp))
        
        return mus
    