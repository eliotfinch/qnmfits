import numpy as np
import qnm as qnm_loader

from scipy.interpolate import UnivariateSpline
from pathlib import Path
from urllib.request import urlretrieve

import os
import h5py


def download_cook_data():
    """
    Download data for the n=8 and n=9 QNMs from 
    https://zenodo.org/records/10093311, and store in the qnmfits/Data
    directory.
    """

    data_dir = Path(__file__).parent / 'Data'
    data_dir.mkdir(parents=True, exist_ok=True)

    for n in [8,9]:

        download_url = f'https://zenodo.org/records/10093311/files/KerrQNM_{n:02}.h5?download=1'
        file_path = data_dir / f'KerrQNM_{n:02}.h5'

        if not file_path.exists():
            print(f'Downloading KerrQNM_{n:02}.h5...')
            urlretrieve(download_url, file_path)
        else:
            print(f'KerrQNM_{n:02}.h5 already downloaded.')


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

        # The method used by the qnm package breaks down for certain modes that
        # approach the imaginary axis (perhaps most notably, the (2,2,8) mode).
        # We load data for these modes separately, computed by Cook & 
        # Zalutskiy.

        data_dir = Path(__file__).parent / 'Data'

        # Dictionary to store the mode data, using our preferred labelling 
        # convention
        multiplet_data = {}

        # A list of multiplet modes
        self.multiplet_list = []

        self.download_check = {}

        for n in [8,9]:

            # Keep track of what data has been downloaded (useful for warnings)
            file_path = data_dir / f'KerrQNM_{n:02}.h5'
            self.download_check[n] = file_path.exists()

            if self.download_check[n]:
                file_path = data_dir / f'KerrQNM_{n:02}.h5'
                with h5py.File(file_path, 'r') as f:
                    for m_key in f[f'n{n:02}'].keys():
                        for mode_key in f[f'n{n:02}'][m_key].keys():
                            mode_key_split = [element.strip('{}') for element in mode_key.split(',')]
                            if len(mode_key_split) == 4:
                                # If there are four indices for this mode, then
                                # we have a "multiplet". We use the convention
                                # of labelling this as two different overtones.
                                l = int(mode_key_split[0])
                                m = int(mode_key_split[1])
                                n_label = mode_key_split[2] + mode_key_split[3]

                                if (l,m,n) not in self.multiplet_list:
                                    self.multiplet_list.append((l,m,n))

                                multiplet_data[(l,m,n_label)] = np.array(f[f'n{n:02}'][m_key][mode_key])

        for key, data in multiplet_data.items():
        
            # Extract relevant quantities
            spins = data[:,0]
            real_omega = data[:,1]
            imag_omega = data[:,2]
            real_A = data[:,3]
            imag_A = data[:,4]
            real_mu = data[:,5::2]
            imag_mu = data[:,6::2]

            # Interpolate omegas
            real_omega_interp = UnivariateSpline(
                spins, real_omega, s=0
                )
            
            imag_omega_interp = UnivariateSpline(
                spins, imag_omega, s=0
                )
            
            # Interpolate angular separation constants
            real_A_interp = UnivariateSpline(
                spins, real_A, s=0             )
            
            imag_A_interp = UnivariateSpline(
                spins, imag_A, s=0             )
            
            # Interpolate mus
            mu_interp = []
            
            for real_mu, imag_mu in zip(real_mu.T, imag_mu.T):
                
                real_mu_interp = UnivariateSpline(
                    spins, real_mu, s=0                  )
                    
                imag_mu_interp = UnivariateSpline(
                    spins, imag_mu, s=0                  )
                
                mu_interp.append((real_mu_interp, imag_mu_interp))

            # Add these interpolated functions to the frequency_funcs dictionary
            self.interpolated_qnm_funcs[key] = [
                (real_omega_interp, imag_omega_interp), mu_interp]
            
            # Add an entry to the qnm_funcs dictionary that mimics the qnm package
            # behaviour
            
            def qnm_func_placeholder(chif, store=True):
                
                omega_interp = self.interpolated_qnm_funcs[key][0]
                omega = omega_interp[0](chif) + 1j*omega_interp[1](chif)
                
                A = real_A_interp(chif) + 1j*imag_A_interp(chif)
                
                mu_interp = self.interpolated_qnm_funcs[key][1]
                mu = [mu_interp_i[0](chif) + 1j*mu_interp_i[1](chif) for mu_interp_i in mu_interp]
                
                return omega, A, mu
            
            self.qnm_funcs[key] = qnm_func_placeholder
        
        
    def interpolate(self, l, m, n):
        
        qnm_func = self.qnm_funcs[l,m,n]
        
        # Extract relevant quantities
        spins = qnm_func.a
        real_omega = np.real(qnm_func.omega)
        imag_omega = np.imag(qnm_func.omega)
        all_real_mu = np.real(qnm_func.C)
        all_imag_mu = np.imag(qnm_func.C)

        # Interpolate omegas
        real_omega_interp = UnivariateSpline(
            spins, real_omega, s=0
            )
        
        imag_omega_interp = UnivariateSpline(
            spins, imag_omega, s=0
            )
        
        # Interpolate mus
        mu_interp = []
        
        for real_mu, imag_mu in zip(all_real_mu.T, all_imag_mu.T):
            
            real_mu_interp = UnivariateSpline(
                    spins, real_mu, s=0
                    )
                
            imag_mu_interp = UnivariateSpline(
                spins, imag_mu, s=0
                )
            
            mu_interp.append((real_mu_interp, imag_mu_interp))

        # Add these interpolated functions to the frequency_funcs dictionary
        self.interpolated_qnm_funcs[l,m,n] = [
            (real_omega_interp, imag_omega_interp), mu_interp]
        
    def omega(self, l, m, n, sign, chif, Mf=1):
        """
        Return a complex frequency, :math:`\omega_{\ell m n}(M_f, \chi_f)`,
        for a particular mass, spin, and mode. One or both of chif and Mf can
        be array_like, in which case an ndarray of complex frequencies is
        returned.
        
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
            
        chif : float or array_like
            The dimensionless spin magnitude of the black hole.
            
        Mf : float or array_like, optional
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
            
        Returns
        -------
        complex or ndarray
            The complex QNM frequency or frequencies of length 
            max(len(chif), len(Mf)).
        """
        # Load the correct qnm based on the type we want
        m *= sign
        
        # Test if the qnm function has been loaded for the requested mode
        if (l,m,n) not in self.qnm_funcs:
            self.qnm_funcs[l,m,n] = qnm_loader.modes_cache(-2, l, m, n)
            
        if type(chif) in [float, np.float64]:
            omega, A, mu = self.qnm_funcs[l,m,n](chif, store=True)
            
        else:
            # Test if the interpolated qnm function has been created (we create 
            # our own interpolant so that we can evaluate the frequencies for 
            # all spins simultaneously)
            if (l,m,n) not in self.interpolated_qnm_funcs:
                self.interpolate(l,m,n)
                
            omega_interp = self.interpolated_qnm_funcs[l,m,n][0]
            omega = omega_interp[0](chif) + 1j*omega_interp[1](chif)
            
        # Use symmetry properties to get the mirror mode, if requested
        if sign == -1:
            omega = -np.conjugate(omega)
        
        return omega/Mf
    
    def omega_list(self, modes, chif, Mf=1):
        """
        Return a frequency list, containing frequencies corresponding to each
        mode in the modes list (for a given mass and spin).
        
        Parameters
        ----------            
        modes : array_like
            A sequence of (l,m,n,sign) tuples to specify which QNMs to load 
            frequencies for. For nonlinear modes, the tuple has the form 
            (l1,m1,n1,sign1,l2,m2,n2,sign2,...).
            
        chif : float or array_like
            The dimensionless spin magnitude of the final black hole.
            
        Mf : float or array_like, optional
            The mass of the final black hole. See the qnm.omega docstring for
            details on units. The default is 1.
            
        Returns
        -------
        list
            The list of complex QNM frequencies.
        """
        # For each mode, call the qnm function and append the result to the
        # list
        
        # Code for linear QNMs:
        # return [self.omega(l, m, n, sign, chif, Mf) for l, m, n, sign in modes]
        
        # Code for nonlinear QNMs:
        return [
            sum([self.omega(l, m, n, sign, chif, Mf) 
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
        #         sum_list.append(self.omega(l, m, n, sign, chif, Mf))
        #     return_list.append(sum(sum_list))
        # return return_list
        
    def mu(self, l, m, lp, mp, nprime, sign, chif):
        """
        Return a spherical-spheroidal mixing coefficient, 
        :math:`\mu_{\ell m \ell' m' n'}(\chi_f)`, for a particular spin and 
        mode combination. The indices (l,m) refer to the spherical harmonic. 
        The indices (l',m',n') refer to the spheroidal harmonic. The spin chif
        can be a float or array_like.

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
            An integer with value +1 or -1, to indicate the sign of the real
            part of the QNM frequency. If the mixing coefficient associated
            with a -1 QNM (i.e. a mirror mode) is requested, then symmetry 
            properties are used for the calculation.
            
        chif : float or array_like
            The dimensionless spin magnitude of the final black hole.

        Returns
        -------
        complex or ndarray
            The spherical-spheroidal mixing coefficient.
        """
        # There is no overlap between different values of m
        if mp != m:
            return 0
        
        # Load the correct qnm based on the type we want
        m *= sign
        mp *= sign
        
        # Our functions return all mixing coefficients with the given 
        # (l',m',n'), so we need to index it to get the requested l
        if abs(m) > 2:
            index = l - abs(m)
        else:
            index = l - 2
        
        # Test if the qnm function has been loaded for the requested mode
        if (lp,mp,nprime) not in self.qnm_funcs:
            self.qnm_funcs[lp,mp,nprime] = qnm_loader.modes_cache(
                -2, lp, mp, nprime)
            
        if type(chif) in [float, np.float64]:
            
            # Access the relevant functions from the qnm_funcs dictionary, and 
            # evaluate at the requested spin. Storing speeds up future 
            # evaluations.
            omega, A, mu = self.qnm_funcs[lp,mp,nprime](chif, store=True)
            mu = mu[index]
            
        else:
            
            if (lp,mp,nprime) not in self.interpolated_qnm_funcs:
                self.interpolate(lp,mp,nprime)
                
            mu_interp = self.interpolated_qnm_funcs[lp,mp,nprime][1][index]
            mu = mu_interp[0](chif) + 1j*mu_interp[1](chif)
            
        # Use symmetry properties to get the mirror mixing coefficient, if 
        # requested
        if sign == -1:
            mu = (-1)**(l+lp)*np.conjugate(mu)
            
        return mu
        
    def mu_list(self, indices, chif):
        """
        Return a list of mixing coefficients, for all requested indices. See
        the qnm.mu() docstring for more details.
        
        Parameters
        ----------
        indices : array_like
            A sequence of (l,m,l',m',n',sign) tuples specifying which mixing 
            coefficients to return.
            
        chif : float
            The dimensionless spin magnitude of the final black hole.

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
            mus.append(self.mu(l, m, lp, mp, nprime, sign, chif))
        
        return mus
    