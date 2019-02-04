#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:31:36 2018

@author: nico
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:20:41 2018

This file contains usefull functions that are used over and over again in the 
jupyter notebook so it just makes a lot of sense to have them sorted here.
The pipeline should be to first load the parameters with the function 
load_parameters() and then change them the entries of the dictionary in the 
notebook. Then the full dict is passed to the functions. This avoids global 
variables in the notebook.
@author: nico
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
#from pypet import Environment, cartesian_product, Parameter
import numba
import scipy.io
#import h5py
from scipy.signal import find_peaks, hilbert, welch
from scipy.ndimage.filters import gaussian_filter1d
from IPython.display import display
import pandas as pd
#pd.options.display.max_rows = 100
from loadParameters import get_regions


def timeIntegrationOU(params, randomseed=0, transinit=None, use_balanced_sc=False):
    '''
    Integrate the FHN equations including noise and coupling with RK2 scheme using @numba 
    and return the trajectories for the potential and the recovery variable.
    Equations:
    du_i/dt = - alpha u_i^3 + beta u_i^2 + gamma u_i - w + I + K * SC_ij*u_j(t-c*DM_ij) + sigma*noise 
    dw_i/dt = u_i - delta - epsilon w_i / tau
    
    Args:
        params: Parameter dictionary obtained by load_parameters() + modifications
        Important parameters for the integration are:
            init: (u0min, u0max, w0min, w0max): Limits for randomly drawn initial conditions for every node [mV]
            randominitstd, only for init=(0,0,0,0): initialize around the FP with gaussian noise and this std
            alpha, beta, gamma: FHN node parameters, determine du/dt
            delta, epsilon, tau: FHN node parameters, determine dw/dt            
            I: Constant background input 
            K: >0, Global coupling strength
            sigma: >0, Variance of the additive noise
            c: >=0, Transmission speed, converts the connection length in DM (DTI data) to a delay, no delay for c=0
            dt: Timestep for simulation [ms]
            duration: Total simulated time [ms]
            globalN:  >=0, Determines number of uniform SC, if 0 use DTI data for connectivity between nodes 
            randomseed: for reproducibility
            dif: Either using diffusive or additive coupling 
    ''' 
    # load parameters from dictionary
    N = params['N']
    if use_balanced_sc:
        SC = params['SC_lr']
    else:
        SC = params['SC']
    DM = params['DM']
    dt = params['dt']
    duration = params['duration']
    I = params['I']
    c = params['c']
    K = params['K']
    sigma = params['sigma']
    init = params['init']
    randominitstd = params['randominitstd']
    dif = params['dif']
   
    # FHN parameters
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    epsilon = params['epsilon']
    tau = params['tau']
    
    # OU parameters
    ou_mean = params['ou_mean']  
    ou_sigma = params['ou_sigma']  
    ou_tau = params['ou_tau']  
    
    # delay matrix:
    if c == 0:
        DM = np.zeros((N, N))
    elif c > 0:
        DM = DM / c / dt # length divided by transmission speed is delay, bring in units of timesteps
    else:
        raise Exception('Parameter c for signal speed must be positive!')
    maxDelay = int(np.max(np.ceil(DM)))
    if (maxDelay)>(duration/dt):
        raise Exception('Simulation time is not long enough for such a low transmission speed, choose higher c or much higher duration')
    # return arrays
    ts = np.arange(0, duration, dt)
    xs = np.zeros((N, len(ts)))
    ys = np.zeros((N, len(ts)))
    
    if randomseed > 0:
        np.random.seed(randomseed)
    # initial conditions, also: pass them to numba for more flexibility
    if init==(99,99,99,99): # !!!CAUTION!!! that was for (0,0,0,0) in timeIntRK4
        # initialize with fixpoint, u_nullc = w_nullc
        P = np.poly1d([-alpha, beta, gamma-1/epsilon, I+delta/epsilon], variable="x")
        # take only the real root
        u_fp = np.real(P.r[np.isreal(P.r)])[0]
        w_fp = (u_fp-delta)/epsilon
        x_init = np.ones(N)*u_fp + np.random.normal(0,randominitstd,N)
        y_init = np.ones(N)*w_fp + np.random.normal(0,randominitstd,N)
    else:
        x_init = np.random.uniform(init[0], init[1], N)
        y_init = np.random.uniform(init[2], init[3], N)
    if maxDelay > 0:
        xs[:,:maxDelay] = np.tile(x_init,(maxDelay,1)).T
        ys[:,:maxDelay] = np.tile(y_init,(maxDelay,1)).T
    else:
        xs[:,0] = x_init
        ys[:,0] = y_init
    # DONT prepare noise array with sigma = 1 for every node for all timesteps
    #noise = np.random.normal(size=(N, len(ts)), scale=dt)
    # instead: generate random number in calculation of ou_noise[n] (in numba)


    # OU process initialization
    ou_noise = np.zeros((N,))

    # actual integration is done with numba for speedup
    if transinit == None:
        xs, ys = timeIntegrationOUNumba(dt, duration, 
                                        N, SC, DM, maxDelay,
                                        alpha, beta, gamma, delta, epsilon, tau,
                                        I, K, 
                                        ou_mean, ou_sigma, ou_tau, ou_noise,
                                        ts, xs, ys, x_init, y_init, dif)
    else:
        prepar = params.copy()
        prepar[transinit[0]] = transinit[1]
        I1 = prepar['I']
        K1 = prepar['K']
        ou_mean1 = prepar['ou_mean']  
        ou_sigma1 = prepar['ou_sigma']  
        ou_tau1 = prepar['ou_tau']  
        tbreak = int(transinit[2]/dt)
# simulate specified number of steps with "initialization parameters"
        xs1, ys1 = timeIntegrationOUNumba(dt, transinit[2], 
                                        N, SC, DM, maxDelay,
                                        alpha, beta, gamma, delta, epsilon, tau,
                                        I1, K1, 
                                        ou_mean1, ou_sigma1, ou_tau1, ou_noise,
                                        ts[:tbreak], xs[:,:tbreak], ys[:,:tbreak], x_init, y_init, dif)
# use the last values x/ys1[:,-1] to initialize x/ys and simulate ramaining steps with "target parameters"
        xs2, ys2 = timeIntegrationOUNumba(dt, duration-transinit[2], 
                                        N, SC, DM, maxDelay,
                                        alpha, beta, gamma, delta, epsilon, tau,
                                        I, K, 
                                        ou_mean, ou_sigma, ou_tau, ou_noise,
                                        ts[tbreak:], xs[:,tbreak:], ys[:,tbreak:], xs1[:,-1], ys1[:,-1], dif)
        xs = np.hstack((xs1,xs2))
        ys = np.hstack((ys1,ys2))
    
    return ts, xs, ys, x_init, y_init, maxDelay
@numba.njit(locals = {'idxX': numba.int64, 'idxY':numba.int64, 'idx1':numba.int64, 'idy1':numba.int64})
def timeIntegrationOUNumba(dt, duration, 
                           N, SC, DM, maxDelay,
                           alpha, beta, gamma, delta, epsilon, tau,
                           I, K, # c is already included in DM
                           ou_mean, ou_sigma, ou_tau, ou_noise,
                           ts, xs, ys, x_init, y_init, dif):
    # load initial values (valid in any case! for delay, this value is just stretched out until maxdelay)
    sqrt_dt = np.sqrt(dt)
    x = x_init.copy()
    y = y_init.copy()
    
    for t in range(maxDelay, len(ts)):  # start from max delay here - only consider interesting time steps!
        for n in range(N):             # all nodes
            x_ext = 0  # no y_ext since sum in FHN is only in u term
            for i in range(N):         # get input of every other node
                if dif==True:
                    x_ext = x_ext + SC[i, n] * (xs[i, int(np.round(t-DM[i,n]))]-x[n]) # if useDM false -> DM=0 -> doesnt matter
                else: 
                    x_ext = x_ext + SC[i, n] * xs[i, int(np.round(t-DM[i,n]))]  # transmission speed kappa (here: c) already in DM (s.o.)
       
            ou_noise[n] = ou_noise[n] + (ou_mean - ou_noise[n]) * dt / ou_tau + ou_sigma * sqrt_dt * np.random.normal()

            # update FHN equations
            x_k1 = - alpha * x[n]**3 + beta * x[n]**2 + gamma * x[n] - y[n] + K * x_ext + ou_noise[n] + I
            y_k1 = (x[n] - delta - epsilon*y[n])/tau
            x_k2 = - alpha * (x[n]+0.5*dt*x_k1)**3 + beta * (x[n]+0.5*dt*x_k1)**2 + gamma * (x[n]+0.5*dt*x_k1) - (y[n]+0.5*dt*y_k1) + K * x_ext + ou_noise[n] + I
            y_k2 = ((x[n]+0.5*dt*x_k1) - delta - epsilon*(y[n]+0.5*dt*y_k1))/tau
            x_k3 = - alpha * (x[n]+0.5*dt*x_k2)**3 + beta * (x[n]+0.5*dt*x_k2)**2 + gamma * (x[n]+0.5*dt*x_k2) - (y[n]+0.5*dt*y_k2) + K * x_ext + ou_noise[n] + I
            y_k3 = ((x[n]+0.5*dt*x_k2) - delta - epsilon*(y[n]+0.5*dt*y_k2))/tau
            x_k4 = - alpha * (x[n]+1.0*dt*x_k3)**3 + beta * (x[n]+1.0*dt*x_k3)**2 + gamma * (x[n]+1.0*dt*x_k3) - (y[n]+1.0*dt*y_k3) + K * x_ext + ou_noise[n] + I
            y_k4 = ((x[n]+1.0*dt*x_k3) - delta - epsilon*(y[n]+1.0*dt*y_k3))/tau
            
            ### update x_n
            x[n] = x[n] + 1./6.*(x_k1+2*x_k2+2*x_k3+x_k4) * dt
            y[n] = y[n] + 1./6.*(y_k1+2*y_k2+2*y_k3+y_k4) * dt
            
            ### save state
            xs[n,t+1] = x[n]
            ys[n,t+1] = y[n]

    return xs, ys


def timeIntegrationRK4(params, randomseed=0, use_balanced_sc=False):
    '''
    Integrate the FHN equations including noise and coupling with RK2 scheme using @numba 
    and return the trajectories for the potential and the recovery variable.
    Equations:
    du_i/dt = - alpha u_i^3 + beta u_i^2 + gamma u_i - w + I + K * SC_ij*u_j(t-c*DM_ij) + sigma*noise 
    dw_i/dt = u_i - delta - epsilon w_i / tau
    
    Args:
        params: Parameter dictionary obtained by load_parameters() + modifications
        Important parameters for the integration are:
            init: (u0min, u0max, w0min, w0max): Limits for randomly drawn initial conditions for every node [mV]
            randominitstd, only for init=(0,0,0,0): initialize around the FP with gaussian noise and this std
            alpha, beta, gamma: FHN node parameters, determine du/dt
            delta, epsilon, tau: FHN node parameters, determine dw/dt            
            I: Constant background input 
            K: >0, Global coupling strength
            sigma: >0, Variance of the additive noise
            c: >=0, Transmission speed, converts the connection length in DM (DTI data) to a delay, no delay for c=0
            dt: Timestep for simulation [ms]
            duration: Total simulated time [ms]
            globalN:  >=0, Determines number of uniform SC, if 0 use DTI data for connectivity between nodes 
            randomseed: for reproducibility
            dif: Either using diffusive or additive coupling 
    ''' 
    # load parameters from dictionary
    N = params['N']
    if use_balanced_sc:
        SC = params['SC_lr']
    else:
        SC = params['SC']
    DM = params['DM']
    dt = params['dt']
    duration = params['duration']
    I = params['I']
    c = params['c']
    K = params['K']
    sigma = params['sigma']
    init = params['init']
    randominitstd = params['randominitstd']
    dif = params['dif']
   
    # FHN parameters
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    delta = params['delta']
    epsilon = params['epsilon']
    tau = params['tau']
    
    # delay matrix:
    if c == 0:
        DM = np.zeros((N, N))
    elif c > 0:
        DM = DM / c / dt # length divided by transmission speed is delay, bring in units of timesteps
    else:
        raise Exception('Parameter c for signal speed must be positive!')
    maxDelay = int(np.max(np.ceil(DM)))
    if (maxDelay)>(duration/dt):
        raise Exception('Simulation time is not long enough for such a low transmission speed, choose higher c or much higher duration')
    # return arrays
    ts = np.arange(0, duration, dt)
    xs = np.zeros((N, len(ts)))
    ys = np.zeros((N, len(ts)))
    
    if randomseed > 0:
        np.random.seed(randomseed)
    # initial conditions, also: pass them to numba for more flexibility
    if init==(99,99,99,99): # !!!CAUTION!!! that was for (0,0,0,0) in timeIntRK4
        # initialize with fixpoint, u_nullc = w_nullc
        P = np.poly1d([-alpha, beta, gamma-1/epsilon, I+delta/epsilon], variable="x")
        # take only the real root
        u_fp = np.real(P.r[np.isreal(P.r)])[0]
        w_fp = (u_fp-delta)/epsilon
        x_init = np.ones(N)*u_fp + np.random.normal(0,randominitstd,N)
        y_init = np.ones(N)*w_fp + np.random.normal(0,randominitstd,N)
    else:
        x_init = np.random.uniform(init[0], init[1], N)
        y_init = np.random.uniform(init[2], init[3], N)
    if maxDelay > 0:
        xs[:,:maxDelay] = np.tile(x_init,(maxDelay,1)).T
        ys[:,:maxDelay] = np.tile(y_init,(maxDelay,1)).T
    else:
        xs[:,0] = x_init
        ys[:,0] = y_init
    # prepare noise vector with sigma = 1, factor sqrt_dt to make noise independent of timestep 
    noise = np.random.standard_normal(size=(N, len(ts))) / np.sqrt(dt)

    # actual integration is done with numba for speedup
    ts, xs, ys = timeIntegrationRK4Numba(dt, duration, 
                                N, SC, DM, maxDelay,
                                alpha, beta, gamma, delta, epsilon, tau,
                                I, K, sigma, noise,
                                ts, xs, ys, dif)
    
    return ts, xs, ys, x_init, y_init, maxDelay
@numba.njit(locals = {'idxX': numba.int64, 'idxY':numba.int64, 'idx1':numba.int64, 'idy1':numba.int64})
def timeIntegrationRK4Numba(dt, duration, 
                              N, SC, DM, maxDelay,
                              alpha, beta, gamma, delta, epsilon, tau,
                              I, K, sigma, noise, # c is already included in DM
                              ts, xs, ys, dif):
    # load initial values
    x = xs[:,0].copy()
    y = ys[:,0].copy()
    
    for t in range(maxDelay, len(ts)):  # start from max delay here - only consider interesting time steps!
        for n in range(N):             # all nodes
            x_ext = 0  # no y_ext since sum in FHN is only in u term
            for i in range(N):         # get input of every other node
                if dif==True:
                    x_ext = x_ext + SC[i, n] * (xs[i, int(np.round(t-DM[i,n]))]-x[n]) # if useDM false -> DM=0 -> doesnt matter
                else: 
                    x_ext = x_ext + SC[i, n] * xs[i, int(np.round(t-DM[i,n]))]  # transmission speed kappa (here: c) already in DM (s.o.)
            # update FHN equations
            x_k1 = - alpha * x[n]**3 + beta * x[n]**2 + gamma * x[n] - y[n] + K * x_ext + sigma * noise[n,t] + I
            y_k1 = (x[n] - delta - epsilon*y[n])/tau
            x_k2 = - alpha * (x[n]+0.5*dt*x_k1)**3 + beta * (x[n]+0.5*dt*x_k1)**2 + gamma * (x[n]+0.5*dt*x_k1) - (y[n]+0.5*dt*y_k1) + K * x_ext + sigma * noise[n,t] + I
            y_k2 = ((x[n]+0.5*dt*x_k1) - delta - epsilon*(y[n]+0.5*dt*y_k1))/tau
            x_k3 = - alpha * (x[n]+0.5*dt*x_k2)**3 + beta * (x[n]+0.5*dt*x_k2)**2 + gamma * (x[n]+0.5*dt*x_k2) - (y[n]+0.5*dt*y_k2) + K * x_ext + sigma * noise[n,t] + I
            y_k3 = ((x[n]+0.5*dt*x_k2) - delta - epsilon*(y[n]+0.5*dt*y_k2))/tau
            x_k4 = - alpha * (x[n]+1.0*dt*x_k3)**3 + beta * (x[n]+1.0*dt*x_k3)**2 + gamma * (x[n]+1.0*dt*x_k3) - (y[n]+1.0*dt*y_k3) + K * x_ext + sigma * noise[n,t] + I
            y_k4 = ((x[n]+1.0*dt*x_k3) - delta - epsilon*(y[n]+1.0*dt*y_k3))/tau
            
            ### update x_n
            x[n] = x[n] + 1./6.*(x_k1+2*x_k2+2*x_k3+x_k4) * dt
            y[n] = y[n] + 1./6.*(y_k1+2*y_k2+2*y_k3+y_k4) * dt

            ### save state
            xs[n,t+1] = x[n]
            ys[n,t+1] = y[n]

    return ts, xs, ys



def plotSpikeTrain(us, params, ou=False):
    '''
    Spike train as an image.
    '''
    fig, ax = plt.subplots(1,1,figsize=(10,4), dpi=100)
    im = ax.pcolormesh(us, cmap='gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    fig.colorbar(im, cax = cax)
    ax.set_xlabel('Simulated timesteps [dt={}ms]'.format(params['dt']))
    ax.set_ylabel(r'$u(t)$ for each node')
    if ou: ax.set_title(r'Spike trains for $I={}, K={}, c={}, \sigma_O={}, \tau_O={}$'.format(params['I'],params['K'],params['c'],params['ou_sigma'],params['ou_tau']))
    else: ax.set_title(r'Spike trains for $I={}, K={}, \sigma={}, c={}$'.format(params['I'],params['K'],params['sigma'],params['c']))
    plt.show()

def plotPhasespaceNodes(ts, xs, ys, u_init, w_init, nodes, params):
    '''
    Draws the phase space spanned by membrane potential u and the recovery variable w.
    WARNING: It only takes the parameters for a single node, noise is not included.
    Therefore, it only takes the equation: du/dt = eps * g(u) - w + I
    The velocity vector np.sqrt(dU**2 + dW**2) is drawn for every point in the 
    grid defined by u_min, u_max, u_step, w_min, w_max, w_step.
    It plots the FPs and nulclines in the phase space
    '''
    I = params['I']
    alpha=params['alpha']; beta=params['beta']; gamma=params['gamma']
    delta=params['delta']; epsilon=params['epsilon']; tau=params['tau']
    u_min=params['ups_min']; u_max=params['ups_max']; u_step=params['ups_step']
    w_min=params['wps_min']; w_max=params['wps_max']; w_step=params['wps_step']
    uu = np.arange(u_min, u_max, u_step)
    ww = np.arange(w_min, w_max, w_step)
    (UU, WW) = np.meshgrid(uu, ww)
    # implement FHN equations
    dU = -alpha * UU**3 + beta * UU**2 + gamma * UU  - WW + I 
    dW = (UU - delta - epsilon*WW)/tau
    # velocity is vector norm
    vel = np.sqrt(dU**2 + dW**2)
    # plot the phase plane
    plt.figure(figsize=(14,10), dpi=100)
    plt.grid(b=True)
    plt.quiver(UU, WW, dU, dW, vel, alpha=0.2)
    plt.xlabel('U')
    plt.ylabel('W')

    # get fixpoints
    P = np.poly1d([-alpha, beta, gamma-1/epsilon, I+delta/epsilon], variable="x")
    # take only the real root
    u_fp = np.real(P.r[np.isreal(P.r)])
    w_fp = (u_fp-delta)/epsilon
    n = u_fp.shape[0]
    for i in range(n):
        plt.plot(u_fp[i],w_fp[i],'*',markersize=12, label='Fixpoint '+str(i+1))
        print('Fixpoint '+str(i+1)+': u='+ str(np.round(u_fp[i],4))+', w='+ str(np.round(w_fp[i],4)))
    # draw nullclines
    u_nullc = -alpha * uu**3 + beta * uu**2 + gamma * uu + I 
    plt.plot(uu,u_nullc, label='u Nullcline', color='orange')
    w_nullc = (uu-delta)/epsilon
    plt.plot(uu,w_nullc, label='w Nullcline', color='magenta')
    plt.xlim(u_min-u_step, u_max)
    plt.ylim(w_min-w_step, w_max)
    plt.legend(loc=1)
    plt.title(r'Phase plane for $I={}, \alpha={}, \beta={}, \gamma={}, \epsilon={}$'.format(I, alpha, beta, gamma, epsilon))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for i, n in enumerate(nodes):
        x_n = xs[n,:]
        y_n = ys[n,:]
        c = colors[i]
        plt.quiver(x_n[:-1], y_n[:-1], x_n[1:]-x_n[:-1], y_n[1:]-y_n[:-1], scale_units='xy',
                   angles='xy', scale=1, label=r'Trajectory for node {}'.format(n), color=c) 
        plt.plot(u_init[n], w_init[n],'o',markersize=12, color=c, label='Starting point node {}'.format(n), alpha=0.25)

    plt.legend(loc=1)
    plt.show()

def plotSpikeTrainNodes(t, us, nodes, params, show_ms=1000):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    dt = params['dt']
    lim = int(show_ms/dt)
    plt.figure(figsize=(8, 5), dpi=250)
    for i, n in enumerate(nodes):
        x_n = us[n,:]
        plt.plot(t[:lim], x_n[:lim], label='u node {}'.format(n), color=colors[i])
    plt.legend(loc=1)
    plt.xlabel('Time [ms]')
    plt.ylabel('u')
    plt.title(r'Spike trains for $I={}, K={}, \sigma={}, c={}$'.format(params['I'],params['K'],params['sigma'],params['c']))
    plt.show()

def plotSpecNodes(us_ana, nodes, params, show_hz=50):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    n, l = us_ana.shape
    fs = 1000/params['dt']
    fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
    for i, n in enumerate(nodes):
        freqs, Pxx_spec = welch(us_ana[n], fs, 'flattop', nperseg=l, scaling='spectrum')
        ax.semilogy(freqs, np.sqrt(Pxx_spec), label='Spectrum node {}'.format(n), color=colors[i])
    ax.axhline(params['rms_thres'], ls='--', color='magenta', label='Oscillation threshold')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(r'$\sqrt{PS_{max}}$')
    ax.set_title(r'Power spectra  for $I={}, K={}, \sigma={}, c={}$'.format(params['I'],params['K'],params['sigma'],params['c']))
    ax.set_xlim(0,show_hz)
    ax.legend(); plt.show()

def mean_freq_amp(us, params, plothist=True):
    '''
    Calculate the mean potential u_mean, the dominant frequency f_dom, and the fluctuation amplitude u_amp
    for each node of a simulated time series us. 
    The parameter peakprom is a measure for the prominance of the dominant peak in the PDS
    '''
    gausfilter = params['gausfilter']
    means = np.mean(us, axis=1)
    n, l = us.shape
    fs = 1000/params['dt']
    if gausfilter>0:
        us = gaussian_filter1d(us, gausfilter, axis=1)  
    amps = np.max(us[:,l//2:], axis=1)-np.min(us[:,l//2:], axis=1)
    dfreqs = np.zeros(n)
    for n in range(n):
        freqs, Pxx_spec = welch(us[n], fs, 'flattop', nperseg=l, scaling='spectrum')
        if np.sqrt(np.max(Pxx_spec)) > params['rms_thres']:
            dfreqs[n] = freqs[np.argmax(Pxx_spec)]
    if plothist:
        ubins = np.linspace(np.min((np.min(amps),np.min(means))), np.max((np.max(amps), np.max(means))), 10)
        fig, ax = plt.subplots(figsize=(6,4), dpi=200)
        n_m, bins_m, _ = ax.hist(x=means, bins=ubins, color='red', alpha=0.5, rwidth=0.85, label=r'Mean $\bar{u}$')
        ax.grid(axis='y', alpha=0.75)
        ax.set_xlabel('u')
        ax.set_ylabel('Number of Nodes')
        ax.tick_params(axis='x')
        ax2 = ax.twiny()
        n_f, bins_f, _ = ax2.hist(x=dfreqs, bins='auto', color='blue', histtype='bar', alpha=0.5, rwidth=0.85, label='Frequency')
        ax2.set_xlabel('Frequency [Hz]', color='blue')
        ax2.tick_params(axis='x', labelcolor='blue')
        n_a, bins_a, _ = ax.hist(x=amps, bins=ubins, color='green', alpha=0.3, rwidth=0.85, label=r'Amplitude $\Delta u$')
        fig.legend(loc=10)
        plt.show()
    return means, dfreqs, amps

def mean_freq_frms_amp(us, params, plothist=True):
    '''
    Calculate the mean potential u_mean, the dominant frequency f_dom, and the fluctuation amplitude u_amp
    for each node of a simulated time series us. 
    The parameter peakprom is a measure for the prominance of the dominant peak in the PDS
    '''
    gausfilter = params['gausfilter']
    means = np.mean(us, axis=1)
    n, l = us.shape
    fs = 1000/params['dt']
    if gausfilter>0:
        us = gaussian_filter1d(us, gausfilter, axis=1)  
    amps = np.max(us[:,l//2:], axis=1)-np.min(us[:,l//2:], axis=1)
    dfreqs = np.zeros(n); frms = np.zeros(n)
    for n in range(n):
        freqs, Pxx_spec = welch(us[n], fs, 'flattop', nperseg=l, scaling='spectrum')
        frms[n] = np.sqrt(np.max(Pxx_spec))
        dfreqs[n] = freqs[np.argmax(Pxx_spec)]

    if plothist:
        ft = dfreqs.copy()
        ft[frms<params['rms_thres']] = 0
        ubins = np.linspace(np.min((np.min(amps),np.min(means))), np.max((np.max(amps), np.max(means))), 10)
        fig, ax = plt.subplots(figsize=(6,4), dpi=200)
        n_m, bins_m, _ = ax.hist(x=means, bins=ubins, color='red', alpha=0.5, rwidth=0.85, label=r'Mean $\bar{u}$')
        ax.grid(axis='y', alpha=0.75)
        ax.set_xlabel('u')
        ax.set_ylabel('Number of Nodes')
        ax.tick_params(axis='x')
        ax2 = ax.twiny()
        n_f, bins_f, _ = ax2.hist(x=dfreqs, bins='auto', color='lightblue', histtype='bar', alpha=0.5, rwidth=0.85, label='Frequency')
        n_ft, bins_ft, _ = ax2.hist(x=ft, bins='auto', color='navy', histtype='bar', alpha=0.5, rwidth=0.85, label='RMS-thres freq')
        ax2.set_xlabel('Frequency [Hz]', color='blue')
        ax2.tick_params(axis='x', labelcolor='blue')
        n_a, bins_a, _ = ax.hist(x=amps, bins=ubins, color='green', alpha=0.3, rwidth=0.85, label=r'Amplitude $\Delta u$')
        fig.legend(loc=10)
        plt.show()
    return means, dfreqs, frms, amps

def kuramoto(xs_osz, plot=False):
    '''
    Input: xs[f>0], spiketrains that show oscillation
    '''
    if xs_osz.shape[0]==0:
        if plot == True:
            print('No oscillation!')
        return 0, 0
    else:
        analytic_signal = hilbert(xs_osz)
        phases = np.angle(analytic_signal) + np.pi
        rad = np.exp(1.j * phases)
        kur = 1/float(rad.shape[0]) * np.abs( np.sum(rad,axis=0))
        kur_mean = np.mean(kur)
        kur_std = np.std(kur)
        if plot == True:
            plt.figure(figsize=(8, 2), dpi=150)
            plt.plot(kur)
            plt.xlabel('Timesteps [dt]')
            plt.ylabel(r'$\dfrac{1}{N}\sum^N_{j=1} e^{i\theta_j}$')
            plt.title(r'Kuramoto Index: $z = ({} \pm {}), {}$  nodes)'.format(np.round(kur_mean, 4), np.round(kur_std, 4),xs_osz.shape[0]))
            plt.show()
        return kur_mean, kur_std

def show_stats(us, params, sortby='connect'):
    SC = params['SC']; N = params['N']; regions = get_regions(N)
    m_pd, f_pd, frms_pd, a_pd = mean_freq_frms_amp(us, params, plothist=False)
    connect = np.sum(SC, axis=1)
    #cluster = np.where(m_pd>0.35, 'LC', 'low')
    #cluster[m_pd>1.35]='high'
    npeaks = np.zeros(N)
    for i in range(N):
        npeaks[i] = int(find_peaks(us[i], prominence=0.5)[0].size)
    data = np.array((np.round(m_pd, 4), np.round(f_pd, 4), np.round(frms_pd, 4), np.round(a_pd, 4), npeaks, np.round(connect, 4)))
    display(pd.DataFrame(data = data.T,
            index = [regions[n+1]+': Node '+str(n).zfill(2) for n in range(90)],
            columns = ['mean', 'dfreq', 'frms', 'amp', 'npeaks', 'connect']).sort_values(sortby, ascending=False))



def show_sc(SC, norm=True, hubs=False, lr=True):
    np.fill_diagonal(SC, 0)

    if norm:
        SC = SC / np.max(SC) 
    fig, axs = plt.subplots(1,2,figsize=(10,4), dpi=100)
    im0 = axs[0].imshow(SC+0.001, norm=LogNorm())
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im0, cax = cax)
    axs[0].set_title('Structural Connectivity Matrix $SC$')
    axs[0].set_ylabel("Node"); axs[0].set_xlabel("Node")
    connectivity = np.sum(SC,axis=0)
    axs[1].plot(connectivity)
    axs[1].set_title('Sum of connection strength for each node')
    axs[1].set_ylabel("Degree"); axs[1].set_xlabel("Node")
    cons = np.argsort(connectivity)
    plt.figtext(0.5, -0.09, 'Highest degree have nodes {} ({}), {} ({}), and {} ({})'.format(cons[-1],np.round(connectivity[cons[-1]],3),cons[-2],np.round(connectivity[cons[-2]],3),cons[-3],np.round(connectivity[cons[-3]],3)), wrap=True, horizontalalignment='center', fontsize=12)
    plt.figtext(0.5, -0.03, 'Maximum value: {}, average value: {} (tr(SC)={})'.format(np.round(np.max(SC),5), np.round(np.mean(SC),5), np.trace(SC)), wrap=True, horizontalalignment='center', fontsize=12)  
    if hubs:
        fig2, axs = plt.subplots(1,3,figsize=(10,4), dpi=100)
        im0 = axs[0].imshow(SC>0.5)
        axs[0].set_title('$SC>0.5$, Hubs: '+str(np.argsort(np.sum(SC>0.5, axis=0))[::-1][:5]))
        axs[0].set_ylabel("Node"); axs[0].set_xlabel("Node")
        im1 = axs[1].imshow(SC>0.25); axs[1].set_xlabel("Node")
        axs[1].set_title('$SC>0.25$, Hubs: '+str(np.argsort(np.sum(SC>0.25, axis=0))[::-1][:3]))
        im2 = axs[2].imshow(SC>0.1); axs[2].set_xlabel("Node")
        axs[2].set_title('$SC>0.1$, Hubs: '+str(np.argsort(np.sum(SC>0.1, axis=0))[::-1][:3]))

    if lr:
        SC_left = SC[0::2,0::2]
        SC_right= SC[1::2,1::2]
        fig, axs = plt.subplots(1,2,figsize=(10,4), dpi=100)
        im0 = axs[0].imshow(SC_left,vmax=SC.max())
        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax = cax)
        axs[0].set_title('Left hemisphere, sum = {}'.format(np.round(np.sum(SC_left),2)))
        axs[0].set_ylabel("Left Nodes"); axs[0].set_xlabel("Left Nodes")
        axs[1].set_title('Right hemisphere, sum = {}'.format(np.round(np.sum(SC_right),2)))
        axs[1].set_ylabel("Right Nodes"); axs[1].set_xlabel("Right Nodes")
        im1 = axs[1].imshow(SC_right,vmax=SC.max())
        divider = make_axes_locatable(axs[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax = cax)
    
    plt.show()
    
def region_connect(SC):
    N = SC.shape[0]
    regions = get_regions(N)
    display(pd.DataFrame(data = np.sum(SC, axis=1),
            index = [regions[n+1]+': Node '+str(n).zfill(2) for n in range(90)],
            columns = ['connect']).sort_values('connect', ascending=False))




def bifurcation_freq_diag(params, minI,maxI, num=500, RMSthres=0):
    varI = np.linspace(minI, maxI,num)
    maxU = np.zeros(num); minU = np.zeros(num); meanU = np.zeros(num)
    dfreqs = np.zeros(len(varI)); dfreqsRMS = np.zeros(len(varI))
    maxU_12 = np.zeros(num); minU_12 = np.zeros(num); meanU_12 = np.zeros(num)
    dfreqs_12 = np.zeros(len(varI)); dfreqsRMS_12 = np.zeros(len(varI))

    params['duration'] = 100000
    fs = 1000/params['dt']
    for i,I in enumerate(varI):
        params['I'] = I
        t1Node, u1Node, _, _, _,_ = timeIntegrationRK4(params,randomseed=42)
        u1_ana = u1Node[0][100000:]
        maxU[i] = np.max(u1_ana); minU[i] = np.min(u1_ana); meanU[i] = np.mean(u1_ana)
        u1_ana_12 = u1Node[0][10000:20000]
        maxU_12[i] = np.max(u1_ana_12); minU_12[i] = np.min(u1_ana_12); meanU_12[i] = np.mean(u1_ana_12)
        freqs, Pxx_spec = welch(u1_ana, fs, 'flattop', nperseg=u1_ana.shape[0], scaling='spectrum')
        freqs_12, Pxx_spec_12 = welch(u1_ana_12, fs, 'flattop', nperseg=u1_ana_12.shape[0], scaling='spectrum')
        dfreqs[i] = freqs[np.argmax(Pxx_spec)]
        dfreqsRMS[i] = np.sqrt(np.max(Pxx_spec))
        # only count a node as oscillating if RMS amplitude exceeds a threshold
        if dfreqsRMS[i]<RMSthres: 
            dfreqs[i] = 0
        dfreqs_12[i] = freqs_12[np.argmax(Pxx_spec_12)]
        dfreqsRMS_12[i] = np.sqrt(np.max(Pxx_spec_12))
        if dfreqsRMS_12[i]<RMSthres: 
            dfreqs_12[i] = 0

    print(u1_ana.shape, u1_ana_12.shape)

    fig, [ax1,ax2] = plt.subplots(2,1, figsize=(8,8), dpi=200)
    
    ax1.plot(varI, maxU, label=r'$u_{max,100sec}$', alpha=0.5, color='darkblue')
    ax1.plot(varI, minU, label=r'$u_{min,100sec}$', alpha=0.5, color='royalblue')
    ax1.plot(varI, meanU, label=r'$\bar{u}_{100sec}$', alpha=0.5, color='blue')
    ax1.plot(varI, maxU_12, label=r'$u_{max,2sec}$', alpha=0.5, color='darkgreen')
    ax1.plot(varI, minU_12, label=r'$u_{min,2sec}$', alpha=0.5, color='limegreen')
    ax1.plot(varI, meanU_12, label=r'$\bar{u}_{2sec}$', alpha=0.5, color='green')
    ax1.set_xlim(minI,maxI); ax1.set_xlabel(r'$I_{ext}$'); ax1.set_ylabel(r'u')
    ax1.set_title(r'Bifurcation diagrams for $\alpha$= '+str(params['alpha'])+r', $\beta$='+str(params['beta'])+r' $\gamma$='+str(params['gamma'])+r' $\epsilon$='+str(params['epsilon'])+r' $\tau$='+str(params['tau']))
    ax1.legend()
        
    ax2.plot(varI, dfreqs, label=r'$f_{max,100sec}$', color='darkred', alpha=0.5)
    ax2.plot(varI, dfreqs_12, label=r'$f_{max,2sec}$', color='tomato', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('Dominant Freq [Hz]', color='red')
    #ax2.set_title(r'Frequency Dependency')
    ax2.set_xlim(minI,maxI); ax2.set_xlabel(r'$I_{ext}$')    
    ax3 = ax2.twinx()
    ax3.plot(varI, dfreqsRMS, label=r'$\sqrt{PS_{max}}_{100sec}$', color='teal', alpha=0.5)
    ax3.plot(varI, dfreqsRMS_12, label=r'$\sqrt{PS_{max}}_{2sec}$', color='cyan', alpha=0.5)
    ax3.tick_params(axis='y', labelcolor='c')
    ax3.set_ylabel(r'$\sqrt{PS_{max}}$', color='c')
    
    ax2.legend(loc='upper left'); ax3.legend(loc='upper right')
    plt.show()
    return varI, dfreqs
