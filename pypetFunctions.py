#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 16:21:39 2018

@author: nico
"""

from loadParameters import *
from defaultFunctions import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from pypet import Environment, cartesian_product, Parameter, Trajectory


def add_parameters(traj, params, use_balanced_sc=False):
    traj.f_add_parameter_group('simulation', comment = 'simulation parameters')                               
    traj.simulation.dt =  Parameter('dt', params['dt'], comment='Size of simulation timestep [ms], choose so that integration is stable!') 
    traj.simulation.duration = Parameter('duration', params['duration'], comment='total simulation time [ms]')
    
    traj.f_add_parameter_group('globalNetwork', comment = 'global network parameters')
    traj.globalNetwork.I = Parameter('I', params['I'])
    traj.globalNetwork.K = Parameter('K', params['K'])
    traj.globalNetwork.sigma = Parameter('sigma', params['sigma'])
    traj.globalNetwork.c = Parameter('c', params['c'])
    traj.globalNetwork.ou_mean = Parameter('ou_mean', params['ou_mean'])
    traj.globalNetwork.ou_sigma = Parameter('ou_sigma', params['ou_sigma'])
    traj.globalNetwork.ou_tau = Parameter('ou_tau', params['ou_tau'])
    if use_balanced_sc: 
        traj.globalNetwork.SC = Parameter('SC', params['SC_lr'])
    else:
        traj.globalNetwork.SC = Parameter('SC', params['SC'])
    traj.globalNetwork.DM = Parameter('DM', params['DM'])
    traj.globalNetwork.N = Parameter('N', params['N'])
    traj.globalNetwork.dif = Parameter('dif', params['dif'])
    
    traj.f_add_parameter_group('localModel', comment = 'local model parameters')    
    traj.localModel.alpha = Parameter('alpha', params['alpha'])
    traj.localModel.beta = Parameter('beta', params['beta'])
    traj.localModel.gamma = Parameter('gamma', params['gamma'])
    traj.localModel.delta = Parameter('delta', params['delta'])
    traj.localModel.epsilon = Parameter('epsilon', params['epsilon'])
    traj.localModel.tau = Parameter('tau', params['tau'])
    traj.localModel.init = Parameter('init', params['init'])
    traj.localModel.randominitstd = Parameter('randominitstd', params['randominitstd'], comment='add gaussian noise with std on initial values')

    traj.f_add_parameter_group('evaluation', comment = 'threshold/analysis parameter for eval')     
    traj.evaluation.u_thres = Parameter('u_thres', params['u_thres'], comment='threshold for HS/LS when no oscillation')
    traj.evaluation.rms_thres = Parameter('rms_thres', params['rms_thres'], comment='threshold for the significance of a freq amplitude')
    traj.evaluation.gausfilter = Parameter('gausfilter', params['gausfilter'], comment='if >0: filter us before analysing')


def simulate(traj, anatime=2000, retAll=False, retSpiketrain=False, retAnalyse=True, retFullspiketrain=False):
    '''
    Run time simulation in pypet for specified parameters.
    transientTime [ms] determines the timespan that is used for the analysis (xs_ana), default: last second
    return: ts_ana, xs_ana, kur, kurstd, nLC
    '''
    params = traj.parameters.f_to_dict(short_names=True, fast_access=True) 
    ts, xs, _, _,_,_ = timeIntegrationOU(params)
    anasteps = int(anatime/params['dt'])
    xs_ana = xs[:,-anasteps:]; ts_ana = ts[-anasteps:]
    m, f, frms, a = mean_freq_frms_amp(xs_ana, params, plothist=False)
    thres_f = f.copy()
    thres_f[frms<params['rms_thres']] = 0
    maxfreq = np.max(thres_f)
    kur, kurstd = kuramoto(xs_ana[thres_f>0], plot=False)
    nLC = np.count_nonzero(thres_f)
    nHS=0; nLS=0
    for i in range(m.size):
        if (m[i]>params['u_thres'] and thres_f[i]<2): 
            nHS += 1
        if (m[i]<params['u_thres'] and thres_f[i]<2): 
            nLS += 1
    if (nLS+nHS+nLC) != params['N']:
        print('!!!!! ComputationERROR: nLS+nHS+nLC is different from N !!!!!')
        
    if retAll:
        traj.f_add_result('results.$', ts=ts_ana, xs=xs_ana, m=m, f=f, frms=frms, a=a, 
                          kur=kur, kurstd=kurstd, nLC=nLC, nHS=nHS, nLS=nLS, maxfreq=maxfreq)
        return ts_ana, xs_ana, m, f, frms, a, kur, kurstd, nLS, nLC, nHS, maxfreq
    if retSpiketrain:
        traj.f_add_result('results.$', ts=ts_ana, xs=xs_ana)
        return ts_ana, xs_ana
    if retAnalyse:
        traj.f_add_result('results.$', m=m, f=f, frms=frms, a=a, 
                          kur=kur, kurstd=kurstd, nLC=nLC, nHS=nHS, nLS=nLS, maxfreq=maxfreq)
        return m, f, frms, a, kur, kurstd, nLS, nLC, nHS, maxfreq
    if retFullspiketrain:
        traj.f_add_result('results.$', ts=ts, xs=xs)
        return ts, xs

