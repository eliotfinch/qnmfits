import numpy as np
import spherical

from .Base import BaseClass


class Custom(BaseClass):
    """
    A class to hold data for any waveform that has been decomposed into 
    spherical-harmonic modes. A metadata dictionary with a remnant mass and 
    spin must be provided, which are used in the flux calculations and frame
    transformations.

    Parameters
    ----------
    times : array_like
        The times associated with the data.
    
    data_dict : dict
        Data decomposed into spherical-harmonic modes. This should have keys 
        (l,m) and array_like data of length times.
        
    metadata : dict
        A dictionary of metadata associated with the data. Required keys are:
            
            - 'remnant_mass'
                float : the mass of the remnant BH, in units of the total 
                binary mass
            - 'remnant_dimensionless_spin'
                array_like : the dimensionless spin vector of the remnant BH.
        
    ellMax : int, optional
        Maximum ell index for modes to include. All available m indicies 
        for each ell will be included automatically. The default is None, 
        in which case all available modes will be included.
        
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
    """
    
    def __init__(self, times, data_dict, metadata, ellMax=None, zero_time=0, 
                 transform=None):        
        """
        Initialize the class.
        """
        self.times = times
        self.metadata = metadata
        self.ellMax = ellMax
        self.zero_time = zero_time
        
        # Load in key pieces of metadata and set as class attributes
        self.load_metadata()
        
        # If no ellMax is given, we load all the modes
        if self.ellMax == None:
            # The maximum l value in the data
            self.ellMax = max([l for (l,m) in data_dict.keys()])
        
        # Only keep data up to ellMax
        self.h = {}
        for lm in data_dict.keys():
            if lm[0] <= self.ellMax:
                self.h[lm] = data_dict[lm]
        
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
        Store useful quantities from the metadata.
        """
        
        # Quantities at reference time
        # ----------------------------
        ref_keys = {
            'reference_time': 'reference_time',
            'reference_mass1': 'm1',
            'reference_mass2': 'm2',
            'reference_dimensionless_spin1': 'chi1',
            'reference_dimensionless_spin2': 'chi2'
            }
        
        for key in ref_keys.keys():
            if key in self.metadata.keys():
                exec(f'self.{ref_keys[key]} = self.metadata[key]')
        
        if (('reference_mass1' in self.metadata.keys()) & 
            ('reference_mass2' in self.metadata.keys())):
            self.M = self.m1 + self.m2
        
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
        if 'remnant_velocity' in self.metadata.keys():
            self.vf = np.array(self.metadata['remnant_velocity'])
        