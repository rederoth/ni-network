#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:25:57 2018

@author: nico
"""

import numpy as np
import scipy.io


def load_parameters(singleNode=False, globalN=0,
                    SC_filename='./aal90/DTI_CM_average.mat',
                    DM_filename='./aal90/DTI_LEN_average.mat'):
    class struct(object):
        pass

    params = struct()
    #FHN parameters with:
    '''
    du/dt = -alpha u^3 + beta u^2 - gamma u - w + I_{ext}
    dw/dt = 1/tau (u + delta  - epsilon w)
    '''
    params.alpha = 4. # eps in kostova
    params.beta = 4*1.5 # eps(1+lam)
    params.gamma = -4/2 # lam eps
    params.delta = 0.
    params.epsilon = 0.5 # a
    params.tau = 1. 
    
    ### runtime parameters
    params.dt = 0.1  # simulation timestep in ms 
    params.duration = 10000  # Simulation duration in ms

    ### network parameters
    if singleNode:
        N = 1
        params.SC = np.zeros((N, N))
        params.DM = np.zeros((N, N))
    elif globalN>0:
        N = globalN
        SC = np.ones((N,N))
        np.fill_diagonal(SC, 0)
        params.SC = SC
        params.SC_lr = SC
        params.DM = SC

    else:
        if SC_filename[-3:]=='npy':
            SC = np.load(SC_filename)
        # params.SC = scipy.io.loadmat('../interareal/data/ritter_data/group/SCmean.mat')['SC']  # np.ones((N,N)) * 0.2
        else: 
            SC = scipy.io.loadmat(SC_filename)['average']

        N = SC.shape[0]
        np.fill_diagonal(SC, 0)
        
        # this balances the left and right hemisphere in the DTI data
        # WARNING: only works if left and right have alternating index!!!
        SC_lr = SC
        np.fill_diagonal(SC_lr, 0)
        r_sum = np.sum(SC_lr[1::2,1::2])
        l_sum = np.sum(SC_lr[0::2,0::2])
        SC_lr[0::2,0::2] *= r_sum/l_sum
        
        params.SC = SC / np.max(SC) 
        params.SC_lr = SC_lr / np.max(SC_lr)
        if DM_filename[-3:]=='npy':
            params.DM = np.load(DM_filename)
        else:
            params.DM = scipy.io.loadmat(DM_filename)['average']

    params.N = N  # number of nodes

    ### global parameters
    params.I = 0.6 # Background input
    params.K = 0.  # global coupling strength
    params.sigma = 0. # Variance of the additive noise
    params.c = 0.  # signal transmission speed
    
    ### OU Process (external noise)
    params.ou_mean = 0.0
    params.ou_sigma = 0.01
    params.ou_tau = 1.0
    
    ### coupling can be additive (default) or diffusive
    params.dif = False
    
    ### Initialization noise
    params.init = (0,0,0,0) #initialize randomly with u in [0,1] & w in [2,3]
    params.randominitstd = 0 # if init=(0,0,0,0) add gaussian noise with std on initial values
    
    ### for phase space plots
    params.ups_min = -.2
    params.ups_max = 1.2
    params.wps_min = 0.
    params.wps_max = 2.
    params.ups_step = 0.05
    params.wps_step = 0.05
    
    ### for clustering / to determine the dominant frequency
    params.u_thres = 0.5 # used to decide wheter non-spiking node is in high or low state
    params.rms_thres = 0.1 # used to discard domfreqs w very low significance
    params.peakprom = 1e5 # not used anymore...
    params.gausfilter = 0
    params.freqlim = 200 # [Hz], 0-freqlim is region of interest 
    
    
    params_dict = params.__dict__
    return params_dict

def load_parameters_params4(singleNode=False):
    '''
    Modification to default parameter set
    For now, priority is sinusodial oscillation, modified so that dfreq more "realistic"
    + : sinusoidal oscillation
    + : u>0 for all I
    + : convenient parameter spaces
    - : "batman shaped" dominant frequency
    '''
    params = load_parameters(singleNode=singleNode, globalN=0, 
                             SC_filename='./aal90/DTI_CM_average.mat', 
                             DM_filename='./aal90/DTI_LEN_average.mat')
    
    params['alpha'] = 3.
    params['beta'] = 4
    params['gamma'] = -3./2
    params['tau'] = 20
    
    params['u_thres'] = 0.5 # given by bifurcation diagram, central value of LC
    params['rms_thres'] = 0.1 # so that even strong noise (sigma=1.) is still not considered oscillation

    params['init'] = (0,0,0,0) # at FP for single node
    params['randominitstd'] = 0.5 # std of normal dist around FP init
    params['I'] = 0.6 # close to bifurcation point
    
    #only for plotting the phase space...
    params['ups_min']=-.5; params['ups_max']=2.;params['wps_min']=0.; params['wps_max']=4.
    params['wps_step']=0.1; params['ups_step']=0.05
    return params

def get_regions(N):
    if N == 90: regions = {1 : 'Precentral_L',
        2 : 'Precentral_R',
        3 : 'Frontal_Sup_L',
        4 : 'Frontal_Sup_R',
        5 : 'Frontal_Sup_Orb_L',
        6 : 'Frontal_Sup_Orb_R',
        7 : 'Frontal_Mid_L',
        8 : 'Frontal_Mid_R',
        9 : 'Frontal_Mid_Orb_L',
        10 : 'Frontal_Mid_Orb_R',
        11 : 'Frontal_Inf_Oper_L',
        12 : 'Frontal_Inf_Oper_R',
        13 : 'Frontal_Inf_Tri_L',
        14 : 'Frontal_Inf_Tri_R',
        15 : 'Frontal_Inf_Orb_L',
        16 : 'Frontal_Inf_Orb_R',
        17 : 'Rolandic_Oper_L',
        18 : 'Rolandic_Oper_R',
        19 : 'Supp_Motor_Area_L',
        20 : 'Supp_Motor_Area_R',
        21 : 'Olfactory_L',
        22 : 'Olfactory_R',
        23 : 'Frontal_Sup_Medial_L',
        24 : 'Frontal_Sup_Medial_R',
        25 : 'Frontal_Med_Orb_L',
        26 : 'Frontal_Med_Orb_R',
        27 : 'Rectus_L',
        28 : 'Rectus_R',
        29 : 'Insula_L',
        30 : 'Insula_R',
        31 : 'Cingulum_Ant_L',
        32 : 'Cingulum_Ant_R',
        33 : 'Cingulum_Mid_L',
        34 : 'Cingulum_Mid_R',
        35 : 'Cingulum_Post_L',
        36 : 'Cingulum_Post_R',
        37 : 'Hippocampus_L',
        38 : 'Hippocampus_R',
        39 : 'ParaHippocampal_L',
        40 : 'ParaHippocampal_R',
        41 : 'Amygdala_L',
        42 : 'Amygdala_R',
        43 : 'Calcarine_L',
        44 : 'Calcarine_R',
        45 : 'Cuneus_L',
        46 : 'Cuneus_R',
        47 : 'Lingual_L',
        48 : 'Lingual_R',
        49 : 'Occipital_Sup_L',
        50 : 'Occipital_Sup_R',
        51 : 'Occipital_Mid_L',
        52 : 'Occipital_Mid_R',
        53 : 'Occipital_Inf_L',
        54 : 'Occipital_Inf_R',
        55 : 'Fusiform_L',
        56 : 'Fusiform_R',
        57 : 'Postcentral_L',
        58 : 'Postcentral_R',
        59 : 'Parietal_Sup_L',
        60 : 'Parietal_Sup_R',
        61 : 'Parietal_Inf_L',
        62 : 'Parietal_Inf_R',
        63 : 'SupraMarginal_L',
        64 : 'SupraMarginal_R',
        65 : 'Angular_L',
        66 : 'Angular_R',
        67 : 'Precuneus_L',
        68 : 'Precuneus_R',
        69 : 'Paracentral_Lobule_L',
        70 : 'Paracentral_Lobule_R',
        71 : 'Caudate_L',
        72 : 'Caudate_R',
        73 : 'Putamen_L',
        74 : 'Putamen_R',
        75 : 'Pallidum_L',
        76 : 'Pallidum_R',
        77 : 'Thalamus_L',
        78 : 'Thalamus_R',
        79 : 'Heschl_L',
        80 : 'Heschl_R',
        81 : 'Temporal_Sup_L',
        82 : 'Temporal_Sup_R',
        83 : 'Temporal_Pole_Sup_L',
        84 : 'Temporal_Pole_Sup_R',
        85 : 'Temporal_Mid_L',
        86 : 'Temporal_Mid_R',
        87 : 'Temporal_Pole_Mid_L',
        88 : 'Temporal_Pole_Mid_R',
        89 : 'Temporal_Inf_L',
        90 : 'Temporal_Inf_R',}
    else: 
        print('ERROR: No parcellation scheme matches this number of nodes!')
        regions = {}
    return regions
