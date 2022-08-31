import numpy as np
import quaternionic
from scipy.interpolate import InterpolatedUnivariateSpline as spline


class BaseClass:
    """
    A base class which contains all the methods used by the other waveform
    classes.
    """
    
    # Functions to calculate (frame-independent) flux quantities
    # ==========================================================
    # We reperform the hdot calculation after any frame changes. Flux
    # quantities can inform start time and frame transformations so we need
    # them first.
    
    def calculate_hdot(self):
        '''
        Calculate the derivative of the strain. First, interpolate h with a 
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
        Helper function to return zero if either l or m are not allowed. 
        Returns a single mode.
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
    
    
    # Functions to deal with the waveform frame
    # =========================================
    
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
        
        # Recalculate the time derivative of the modes in the new frame
        self.calculate_hdot()
        
        
    def rotate_modes_over_time(self):
        """
        Reperform the spin-weighted spherical harmonic decomposition in a 
        rotated coordinate system where the z-axis is parallel to the 
        instantanious black-hole spin.
        
        The modes in the h dictionary are over-written.
        """
        # Reshape chioft magnitudes to make division possible
        chioft_mag = self.chioft_mag.reshape(len(self.chioft_mag),1)
        
        # Get the changing spin direction
        chioft_norm = self.chioft/chioft_mag
        thetaoft = np.arccos(chioft_norm[:,2])
        phioft = np.arctan2(chioft_norm[:,1], chioft_norm[:,0])
        
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
        
        # Recalculate the time derivative of the modes in the new frame
        self.calculate_hdot()
    
    
    # Functions to calculate properties of the waveform time evolution
    # ================================================================
        
    def calculate_foft(self, method='phase_derivative'):
        """
        Calculate the frequency evolution of each of the waveform modes (up to
        ellMax) in units of cycles/M by the user-requested method. 
        
        The results are stored to the self.foft dictionary. This has keys for 
        each [l,m] mode.

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

        
    # Other helper functions
    # ======================
        
    def project_signal(self, theta, phi):
        """
        Project the signal in the requested position on the (source frame) sky.

        Parameters
        ----------
        theta : float
            The angle between the source-frame z-axis and the line of sight 
            (i.e. the inclination).
            
        phi : float
            The azimuthal angle of the line of sight in the source frame.

        Returns
        -------
        signal : ndarray
            The complex gravitational wave signal.
            
        """
        # Initialize an empty, complex array
        signal = np.zeros_like(self.times, dtype=complex)
            
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
                        