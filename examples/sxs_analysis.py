# Script to perform ringdown analyses on a SXS simulation

import numpy as np
import matplotlib.pyplot as plt

import qnmfits

# SXS simulation setup
# --------------------

ID = 305
transform = [None, 'rotation', 'dynamic_rotation', 'boost'][0]
zero_time = [(2,2), 'Edot', 'common_horizon', None][0]
ellMax = [2,3,None][2]

# Initialize the simulation
sim = qnmfits.Simulation(
    ID, ellMax=ellMax, zero_time=zero_time, transform=transform, 
    lev_minus_highest=0)
sim.print_metadata()

#%% 

# Ringdown fit
# ------------

t0 = 0
T = 100

l_max = 2

# Fixed m
modes = [(l,2,n) for l in range(2,l_max+1) for n in range(1+1)]
hlm_modes = [(l,2) for l in range(2,l_max+1)]

# All modes
# modes = [(l,m,n) for l in range(2,3+1) for m in range(-l,l+1) for n in range(7+1)]
# hlm_modes = [(l,m) for l in range(2,3+1) for m in range(-l,l+1)]

mirror_modes = [[], modes][0]
# modes = []

best_fit = sim.ringdown_fit(
    t0=t0, T=T, modes=modes, mirror_modes=mirror_modes, hlm_modes=hlm_modes)

print(f"Mismatch = {best_fit['mismatch']}")

# Plots
# -----
fig_kw = {'dpi': 300}

if True:

    # Unweighted, shared amplitudes
    sim.plot_mode_amplitudes(
        best_fit['C'], best_fit['mode_labels'], log=False, fig_kw=fig_kw)
    
    # Plots for each spherical harmonic mode
    for lm in hlm_modes:
        
        sim.plot_mode_amplitudes(
            best_fit['weighted_C'][lm], best_fit['mode_labels'], log=False, 
            fig_kw=fig_kw)
        
        sim.plot_ringdown(
            hlm_mode=lm, xlim=[t0-40,T+10], best_fit=best_fit, fig_kw=fig_kw)
        
        peak_value = np.max(abs(sim.h[lm]))
        
        sim.plot_ringdown_modes(
            best_fit,
            hlm_mode=lm,
            xlim=(t0-40,T+10), 
            ylim=(-peak_value*1.1,peak_value*1.1), 
            legend=False,
            fig_kw=fig_kw)
        
    
#%% 

# Mismatch as a function of time
# ------------------------------

t0_array = np.linspace(-20, 50, 50)

# Create figure
fig, ax = plt.subplots(dpi=300)
    
# Create mismatch list
mm_list = sim.mismatch_t0_array(
    t0_array, T_array=T, modes=modes, mirror_modes=mirror_modes, hlm_modes=hlm_modes)

# Add to figure
ax.semilogy(t0_array, mm_list)
    
# Plot limits and labels
ax.set_xlim(t0_array[0], t0_array[-1])
ax.set_xlabel(f'$t_0\ [M]$ [{sim.zero_time_method}]')
ax.set_ylabel('$\mathcal{M}$')


#%% 

# Mismatch as a function of mass and spin
# ---------------------------------------

# Create grid, choosing mass and spin extents
mm_grid = sim.mismatch_M_chi_grid(
    
    # [sim.Mf-0.001, sim.Mf+0.001], [sim.chif_mag-0.001, sim.chif_mag+0.001],
    [sim.Mf-0.1, sim.Mf+0.1], [sim.chif_mag-0.1, sim.chif_mag+0.1],
    # [sim.Mf-0.5, sim.Mf+0.5], [0., 1.], 
    
    res=50, 
    t0=t0, 
    T=T, 
    modes=modes, 
    mirror_modes=mirror_modes, 
    hlm_modes=hlm_modes)

epsilon, delta_M, delta_chi = sim.calculate_epsilon(
    t0=t0, T=T, modes=modes, mirror_modes=mirror_modes, hlm_modes=hlm_modes, 
    method=['Nelder-Mead','Powell','trust-constr','grid'][0])

sim.plot_mismatch_M_chi_grid(plot_bestfit=True, fig_kw=fig_kw)

print(f'\n\nepsilon = {epsilon}')
print(f'delta_M = {delta_M}')
print(f'delta_chi = {delta_chi}\n')
