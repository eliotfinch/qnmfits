import numpy as np

import spherical
import sxs

from scipy import signal
from tabulate import tabulate

from .Base import BaseClass


class SXS(BaseClass):
    """
    A class to hold the data for a simulation from the SXS catalog.
    
    Parameters
    ----------
    ID : int
        The ID number of the desired SXS simulation.
        
    ellMax : int, optional
        Maximum ell index for modes to include. All available m indicies for
        each ell will be included automatically. The default is None, in which
        case all available modes will be included.
        
    zero_time : optional
        The method used to determine where the time array equals zero. 
        
        Options are:
            
        - time : float
            The time on the default SXS simulation time array where we set t=0.
        - (l,m) : tuple
            The peak of the absolute value of this mode is where t=0.
        - 'norm'
            The time of the peak of total amplitude (e.g. Eq. 2 of 
            https://arxiv.org/abs/1705.07089 ) is used.
        - 'Edot'
            The time of peak energy flux is used.
        - 'common_horizon'
            The time when the common horizon is first detected is used.
          
        The default is 0 (the default simulation zero time).
        
    transform : str or list, optional
        Transformations to apply to the SXS data. Options are:
            
        - 'rotation'
            Reperform the spin-weighted spherical harmonic decomposition in a 
            rotated coordinate system where the z-axis is parallel to the 
            remnant black hole spin.
        - 'dynamic_rotation'
            Reperform the spin-weighted spherical harmonic decomposition in a 
            rotated coordinate system where the z-axis is parallel to the 
            instantanious black hole spin.
            
        The default is None (no tranformations are applied).
        
    lev_minus_highest : int, optional
        The simulation level, relative to the highest available level, to use
        in the analysis. For example, setting this to -1 means the second 
        highest level is used. Default is 0.
        
    extrapolation_order : int, optional
        The extrapolation order to use in the analysis. Available are 2, 3,
        and 4. In addition, using -1 returns the outermost extraction. The 
        default is 2.
    """

    def __init__(self, ID, ellMax=None, zero_time=0, transform=None, 
                 lev_minus_highest=0, extrapolation_order=2):
        """
        Initialize the class.
        """
        self.ID = f'{int(ID):04d}'
        self.ellMax = ellMax
        self.zero_time = zero_time
        self.lev_minus_highest = lev_minus_highest
        self.extrapolation_order = extrapolation_order
        
        # Download the metadata for the simulation (if it hasn't been already).
        # This saves the .json file to the /home/user/.cache/sxs folder,
        # in an appropiately named subfolder. Note, this defaults to the
        # highest available level.
        self.metadata = sxs.load(f'SXS:BBH:{self.ID}/Lev/metadata.json')
        
        # Highest available level
        self.highest_lev = int(self.metadata['simulation_name'][-1])
        
        # The requested level
        self.level = self.highest_lev + self.lev_minus_highest
        
        # Now download the metadata for the requested level, if different from
        # the requested level
        if self.highest_lev != self.level:
            self.metadata = sxs.load(
                f'SXS:BBH:{self.ID}/Lev{self.level}/metadata.json')
        
        # Load in key pieces of metadata and set as class attributes
        self.load_metadata()
        
        # Download the data for the simulation (if it hasn't been already). 
        # This saves the .h5 file to the same place as the metadata json. 
        self.data = sxs.load(
            f'SXS:BBH:{self.ID}/Lev{self.level}/rhOverM', 
            extrapolation_order=extrapolation_order)
        
        # Load data and store to the h dictionary
        self.load_data()
        
        # Frame independent flux quantities
        # ---------------------------------
        
        # Calculate waveform mode time derivatives
        self.calculate_hdot()
        
        # Calculate energy flux
        self.calculate_Moft()
        
        # Calculate angular momentum flux
        self.calculate_chioft()
        
        # Frame transformations
        # ---------------------
        
        # Shift the time array to use the requested zero-time.
        self.time_shift()
        
        # Construct a Wigner object
        self.wigner = spherical.Wigner(self.ellMax)
        
        # Apply tranformations
        if type(transform) != list:
            transform = [transform]
        
        for transformation in transform:
            if transformation == 'rotation':
                self.rotate_modes()
            elif transformation == 'dynamic_rotation':
                self.rotate_modes_over_time()
            elif transformation == 'boost':
                pass
            elif transformation is None:
                pass
            else:
                print('Requested transformation not available.')
        
        # Other interesting quantities
        # ----------------------------
        
        # Calculate the approximate frequency evolution
        self.calculate_foft()
        
        
    def load_metadata(self):
        """
        Read in simulation metadata from the catalog json.
        """
        
        # Quantities at reference time
        # ---------------------------- 
        self.reference_time = self.metadata['reference_time']
        
        self.m1 = self.metadata['reference_mass1']
        self.m2 = self.metadata['reference_mass2']
        self.M = self.m1 + self.m2
        assert abs(self.M-1)<1e-3, 'M not close to one'
        
        self.chi1 = np.array(self.metadata['reference_dimensionless_spin1'])
        self.chi2 = np.array(self.metadata['reference_dimensionless_spin2'])
            
        self.r1 = np.array(self.metadata['reference_position1'])
        self.r1_mag = np.linalg.norm(self.r1)
        self.r2 = np.array(self.metadata['reference_position2'])
        self.r2_mag = np.linalg.norm(self.r2)
        
        self.omega_ref = np.array(self.metadata['reference_orbital_frequency'])
            
        # Common horizon time
        # -------------------
        self.common_horizon_time = self.metadata['common_horizon_time']
            
        # Number of orbits
        # ----------------
        self.Norbits = self.metadata['number_of_orbits']
        
        # Remnant properties
        # ------------------
        self.Mf = self.metadata['remnant_mass']
            
        self.chif = np.array(self.metadata['remnant_dimensionless_spin'])
        self.chif_mag = np.linalg.norm(self.chif)
        
        # Angular coordinates of the final spin vector
        chif_norm = self.chif/self.chif_mag
        self.thetaf = np.arccos(chif_norm[2])
        self.phif = np.arctan2(chif_norm[1], chif_norm[0])
        
        # Kick vector
        self.vf = np.array(self.metadata['remnant_velocity'])
                
        # Derived properties
        # ------------------
        
        # Calculate centre of mass at reference time (useful to check if chip
        # calculation is valid)
        self.com = self.m1*self.r1 + self.m2*self.r2
         
        # Mass ratio
        self.q = self.m1/self.m2
        
        # A1 and A2 (used in chip calculation)
        A1 = 2 + 3/(2*self.q)
        A2 = 2 + (3/2)*self.q
        
        # Orbital angular momentum
        self.L = (self.m1*self.r1_mag**2 + self.m2*self.r2_mag**2)*self.omega_ref
        self.L_norm = self.L/np.linalg.norm(self.L)
        
        # Components of spin angular momenta perpendicular to orbital angular 
        # momentum
        self.S1_perp = self.m1**2*np.linalg.norm(np.cross(self.chi1, self.L_norm))
        self.S2_perp = self.m2**2*np.linalg.norm(np.cross(self.chi2, self.L_norm))
        
        # Components of dimensionless spin parallel to orbital angular momentum
        self.chi1_para = np.dot(self.chi1, self.L_norm)
        self.chi2_para = np.dot(self.chi2, self.L_norm)
        
        # Effective spin parameter
        self.chi_eff = (
            self.m1*self.chi1_para + self.m2*self.chi2_para)/(self.m1 + self.m2)
        
        # Precession spin parameter
        self.Sp = 0.5*(A1*self.S1_perp + A2*self.S2_perp 
                       + abs(A1*self.S1_perp - A2*self.S2_perp))
        
        # Dimensionless precession spin parameter
        self.chip = self.Sp/(A1*self.m1**2)
        
        # Final spin angular momentum
        self.Sf = self.chif*self.Mf**2
        

    def load_data(self):
        """
        Load simulation data from file.
        """
        # Data truncation
        # ---------------
        
        # Work with the 22 mode to truncate data and get time array:
        h22 = self.data[:, self.data.index(2,2)]
        
        # Some simulations contain more orbits than we need; we take the last
        # ~10 orbits for simulations with more than 10 orbits
        if self.Norbits > 10:
            
            # Get peak indices (up to ~merger) using the real part of the 22 
            # mode
            h22_real = h22.data.real[:np.argmax(h22.abs)]
            peak_indices = signal.find_peaks(h22_real)[0]
            
            # We will cut data before the index of the 20th peak from merger,
            # which corresponds to ~10 orbits
            mask_start = peak_indices[-20:][0]
            
        else:
            
            # For simulations with less than 20 orbits, we use all the data
            mask_start = 0
            
        # Load data
        # ---------
        
        # Extract the time column from the data
        self.times = h22.t[mask_start:]

        # Dictionary to store the modes
        self.h = {}
        
        # If no ellMax is given, we load all the modes
        if self.ellMax == None:
            # The maximum l value in the data
            self.ellMax = self.data.ell_max

        # Load data for each mode and store in the h dictionary
        for l in range(2, self.ellMax+1):
            for m in range(-l,l+1):
                self.h[l,m] = np.array(
                    self.data[:, self.data.index(l,m)])[mask_start:]
                
            
    def print_metadata(self):
        """
        Print metadata associated with the waveform.
        """
        print(tabulate([
            ['chi1', self.chi1], 
            ['chi2', self.chi2],
            ['Mf', self.Mf],
            ['chif', self.chif],
            ['vf', self.vf],
            ['q', self.q],
            ['chi_eff', self.chi_eff],
            ['chip', self.chip]
            ]))
        