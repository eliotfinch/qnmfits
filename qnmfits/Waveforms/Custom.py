import numpy as np
import spherical

from .Base import BaseClass


class Custom(BaseClass):
    
    def __init__(self, strain_data, metadata, ellMax=None, zero_time=0, 
                 transform=None):
        """
        Initialize the class.
        """
        self.times = strain_data['times']
        self.h = strain_data
        self.metadata = metadata
        self.ellMax = ellMax
        self.zero_time = zero_time
        
        # Construct a Wigner object
        self.wigner = spherical.Wigner(self.ellMax)
        
        # Load in key pieces of metadata and set as class attributes
        self.load_metadata()
        
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
                
        # Shift the time array
        self.time_shift()
        
    def load_metadata(self):
        """
        Store useful quantities from the metadata.
        """
            
        # Quantities at reference time
        # ---------------------------- 
        self.reference_time = self.metadata['reference_time']
        
        self.m1 = self.metadata['reference_mass1']
        self.m2 = self.metadata['reference_mass2']
        self.M = self.m1 + self.m2
        assert abs(self.M-1) < 1e-3, 'M not close to one'
        
        self.chi1 = np.array(self.metadata['reference_dimensionless_spin1'])
        self.chi2 = np.array(self.metadata['reference_dimensionless_spin2'])
        
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
        