#!/usr/bin/env python
# coding: utf-8

# #  Necessary Packages

# In[1]:


""" Numpy """
import numpy as np

"""Pandas"""
import pandas as pd

"""Matplotlib"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.units as munits
import matplotlib.ticker
from   cycler import cycler
import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm  
import matplotlib as mpl

"""Seaborn"""
import seaborn as sns

""" Wavelets """
import pywt

""" Scipy """
import scipy.io
from scipy.io import savemat

"""Numba"""
from numba import jit, prange

"""Sort files in folder"""
import natsort

""" Load files """
import pickle
import glob
import os

""" Import manual functions """
import sys


#Finds the times where the angle, theta, crosses the threshold    

def final_func(B, av_hours, threshold_angle):
    
   # nindex        = B.index
   # B             = V.reindex(V.index.union(nindex)).interpolate(method='nearest').reindex(nindex)

    B['Bmag']     = np.sqrt(B.Br**2 + B.Bt**2 + B.Bn**2)

    ### Define averaging window
    lag          = (B.index[1]-B.index[0])/np.timedelta64(1,'s')
    av_window    = int(av_hours*3600/lag)

    ### Estimate moving average ###
    B['Br_av'] = B['Br'].rolling(av_window, center=True).mean()
    B['Bt_av'] = B['Bt'].rolling(av_window, center=True).mean()
    B['Bn_av'] = B['Bn'].rolling(av_window, center=True).mean()
        

    B['Br_rms'] = (B['Br']**2).rolling(av_window, center=True).mean()
    B['Bt_rms'] = (B['Bt']**2).rolling(av_window, center=True).mean()
    B['Bn_rms'] = (B['Bn']**2).rolling(av_window, center=True).mean()

    B['Br_var']   = B['Br_rms'] -B['Br_av']**2
    B['Bt_var']   = B['Bt_rms'] -B['Bt_av']**2
    B['Bn_var']   = B['Bn_rms'] -B['Bn_av']**2

    B['Bmag_av']  = np.sqrt(B['Br_av']**2 + B['Bt_av']**2 + B['Bn_av']**2) 

    theta2 = np.arccos((B.Br*B.Br_av+B.Bn*B.Bn_av+B.Bt*B.Bt_av)/(B['Bmag']*B['Bmag_av']))*(180/np.pi) #angle between B and <B>

    time1     = []
    time2     = []
    excur_br  = []
    excur_bt  = []
    excur_bn  = []
    dur       = []
    points    = []
    counts    = []

    time1, time2             = anglefinder(theta2.values, B.index, time1,time2, threshold_angle)

    points, excur_br, counts = estimatedexcur(B.Br_av.values, B.Br.values, time1, time2, B.index, excur_br, points, counts)
    points, excur_bt, counts = estimatedexcur(B.Bt_av.values, B.Bt.values, time1, time2, B.index, excur_bt, points, counts)
    points, excur_bn, counts = estimatedexcur(B.Bn_av.values, B.Bn.values, time1, time2, B.index, excur_bn, points, counts)

    return time1, time2, points, excur_br,excur_bt,excur_bn, counts, B.Br_av.values,B.Bt_av.values, B.Bt_av.values,B.Br_rms.values,B.Br_var.values, B.Bt_rms.values, B.Bt_var.values, B.Bn_rms.values, B.Bn_var.values




def anglefinder(angles, time, test1, test2,threshold):

    index = 0 #set index to zero so I know where I am at during the loop
    while index < len(angles)-1:
        if angles[index] >= threshold: #determine if the angles meet threshold (the angles never are actually 90 so this was the next best thing)
            if angles[index + 1] > threshold and angles[index -1] < threshold: #determine if this is the beginning or end of the switchback
                test1.append(time[index])
            if (angles[index + 1] < threshold and angles[index - 1] > threshold) or index+1 == len(angles)-1: #determine if this is the beginning or end of the switchback 
            #!!!!!!!!!   there is a problem when your last point is still inside a switchback. 
            # i have added a condition to fix that but the same goes for the first point. 
            # You have to take care of what happens whn the angle >90 for index==0. \
            # There was also a typo in "threshhold" that I fixed.
                test2.append(time[index])
        index += 1  
    return (test1,test2)


#Finds the estimated total excursion of the switchback 
def estimatedexcur(mode_val, br, time1, time2, time, excur, points, counts):
    index = 0
    temp1 = [] #Create temporary lists to store the values so I can average them
    temp2 = []
    isb = 0
    while (index < len(time2)) & (index < len(time1)):
        #determine how many minutes each value is to find duration
        val1, = np.where(time == (time1[index]))
        val2, = np.where(time == (time2[index]))
        start = val1[0]
        end = val2[0]
        if (end-start>0):
            for x in range(start, end):
            #This step is to find the magnetic field between the duration
            #so that we can find the max within the duration. The other value
            #is so that we can find the average of the modal magnetic field.
                temp1.append(mode_val[x])
                temp2.append(br[x])
            avg = np.average(temp1) #This could be changed to maybe the lowest modal value.
            high = np.min(temp2)
            #excur.append((high - avg)**2) #The excursion is estimated from subtracting the max from the average.
            excur.append(np.average((np.asarray(temp1)-np.asarray(temp2))**2))

            temp1 = []
            temp2 = []

            isb=isb+1
        index+=1
    counts.append(int(isb))

    return points, excur, counts


@jit(nopython=True)
def slid_av(den,delta_i,den_av):
    i = 0
    while i < len(den):
        if ((i-int(delta_i/2))>=0 and (i+int(delta_i/2)<=len(den)-1)):
            x = np.sum(den[i-int(delta_i/2):i+int(delta_i/2)])/delta_i
            den_av[i] = x
            i += 1
        if i-int(delta_i/2)<0:
            x = np.sum(den[i:i+delta_i])/delta_i
            den_av[i] = x
            i += 1
        if i+int(delta_i/2)>len(den)-1:
            x = np.sum(den[i-delta_i:i])/delta_i
            den_av[i] = x
            i += 1
    return den_av



@jit(nopython=True)
def elimiatesb(br, time1, time2, time, br_nsb, time_nsb):
    index = 0
    del_idx = []
    while index < len(time1):
        #determine how many minutes each value is to find duration
        val1, = np.where(time == (time1[index]))
        val2, = np.where(time == (time2[index]))
        start = val1[0]
        end = val2[0]
        if (end-start>0):
            del_idx.extend(iter(range(start, end)))
        index+=1
    br_nsb = np.delete(br,del_idx)
    time_nsb = np.delete(time,del_idx)
    return br_nsb, time_nsb


@jit(nopython=True)
def slid_av(den,delta_i,den_av):
    i = 0
    while i < len(den):
        if ((i-int(delta_i/2))>=0 and (i+int(delta_i/2)<=len(den)-1)):
            x = np.sum(den[i-int(delta_i/2):i+int(delta_i/2)])/delta_i
            den_av[i] = x
            i += 1
        if i-int(delta_i/2)<0:
            x = np.sum(den[i:i+delta_i])/delta_i
            den_av[i] = x
            i += 1
        if i+int(delta_i/2)>len(den)-1:
            x = np.sum(den[i-delta_i:i])/delta_i
            den_av[i] = x
            i += 1
    return den_av



# Remove gaps from
def remove_big_gaps(big_gaps, B_resampled):
    import datetime
    keys =list(big_gaps.keys())
    """ Now remove the gaps indentified earlier """
    if len(big_gaps)>0:
        for o in range(len(big_gaps)):
            if o%5==0:
                print(f"Completed = {str(100 * o / len(big_gaps))}")
            dt2 = big_gaps.index[o]
            dt1 = big_gaps.index[o]-datetime.timedelta(seconds=big_gaps[keys[0]][o])
            if o==0:
                B_resampled1   = B_resampled[((B_resampled['Starting_Time']<dt1) )  | (B_resampled['Ending_Time']>dt2) ]
            else:
                B_resampled1   = B_resampled1[((B_resampled1['Starting_Time']<dt1) )  | (B_resampled1['Ending_Time']>dt2) ]   

    else:
        B_resampled1 = B_resampled
    return B_resampled1


def identify_gaps_in_timeseries(df, gap_time_threshold):
    
    # First find out what keys does the timeseries have
    keys = list(df.keys())

    # Identify  big gaps in our timeseries ###
    f2          = df.dropna()
    time        = (f2.index.to_series().diff()/np.timedelta64(1, 's'))
    return time[time>gap_time_threshold]



def estimate_quants_particle_data(nindex, in_rtn, df_part, BB, subtract_rol_mean, smoothed = True):
   
    from scipy import constants
    mu_0            = constants.mu_0  # Vacuum magnetic permeability [N A^-2]
    mu0             = constants.mu_0   #
    m_p             = constants.m_p    # Proton mass [kg]
    kb              = constants.k      # Boltzman's constant     [j/K]
    au_to_km        = 1.496e8
    T_to_Gauss      = 1e4

    
    """Define magentic field components"""
    Bx     = BB.values.T[0];  By     = BB.values.T[1];  Bz     = BB.values.T[2]; Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)
    
    if subtract_rol_mean:
        BB[['Br_mean','Bt_mean','Bn_mean']]                   = BB[['Br','Bt','Bn']].rolling('2H', center=True).mean().interpolate()
        df_part[['Vr_mean', 'Vt_mean','Vn_mean', 'np_mean']]  = df_part[['Vr','Vt','Vn', 'np']].rolling('2H', center=True).mean().interpolate()
        #df_part[['np_mean_small']]                            = df_part[['np']].rolling('1min', center=True).mean().interpolate()

    #Estimate median solar wind speed   
    Vth       = df_part.Vth.values;   Vth[Vth < 0] = np.nan; Vth_mean =np.nanmedian(Vth); Vth_std =np.nanstd(Vth);
    
    #Estimate median solar wind speed  
    if in_rtn:
        try:
            Vsw       = np.sqrt(df_part.Vr.values**2 + df_part.Vt.values**2 + df_part.Vn.values**2); Vsw_mean =np.nanmedian(Vsw); Vsw_std =np.nanstd(Vsw);
        except:
            Vsw       = np.sqrt(df_part.Vx.values**2 + df_part.Vy.values**2 + df_part.Vz.values**2); Vsw_mean =np.nanmedian(Vsw); Vsw_std =np.nanstd(Vsw);

    else:
        Vsw       = np.sqrt(df_part.Vx.values**2 + df_part.Vy.values**2 + df_part.Vz.values**2); Vsw_mean =np.nanmedian(Vsw); Vsw_std =np.nanstd(Vsw);
    Vsw[Vsw < 0] = np.nan
    Vsw[np.abs(Vsw) > 1e5] = np.nan

    # estimate mean number density
    Np        = df_part['np'].values
    Np_mean =np.nanmedian(Np); Np_std =np.nanstd(Np);
        
    # Estimate Ion inertial length di in [Km]
    di        = 228/np.sqrt(Np); di[np.log10(di) < -3] = np.nan;  di_mean =np.nanmedian(di); di_std =np.nanstd(di);
    
    # Estimate plasma Beta
    km2m        = 1e3
    nT2T        = 1e-9
    cm2m        = 1e-2
    B_mag       = Bmag * nT2T                              # |B| units:      [T]
    temp        = 1./2 * m_p * (Vth*km2m)**2              # in [J] = [kg] * [m]^2 * [s]^-2
    dens        = Np/(cm2m**3)                            # number density: [m^-3] 
    beta        = (dens*temp)/((B_mag**2)/(2*mu_0))       # plasma beta 
    beta[beta < 0] = np.nan
    beta[np.abs(np.log10(beta))>4] = np.nan # delete some weird data
    beta_mean   = np.nanmedian(beta); beta_std   = np.nanstd(beta);
    
    
    # ion gyro radius
    rho_ci = 10.43968491 * Vth/B_mag #in [km]
    rho_ci[rho_ci < 0] = np.nan
    rho_ci[np.log10(rho_ci) < -3] = np.nan
    rho_ci_mean =np.nanmedian(rho_ci); rho_ci_std =np.nanstd(rho_ci);
 
    ### Define b and v ###
    if in_rtn:
        try:
            Vr, Vt, Vn   = df_part.Vr.values, df_part.Vt.values, df_part.Vn.values
            Br, Bt, Bn   = BB.Br.values, BB.Bt.values, BB.Bn.values
        except:
            Vr, Vt, Vn   = df_part.Vx.values, df_part.Vy.values, df_part.Vz.values
            Br, Bt, Bn   = BB.Bx.values, BB.By.values, BB.Bz.values
    else:
        Vr, Vt, Vn       = df_part.Vx.values, df_part.Vy.values, df_part.Vz.values
        Br, Bt, Bn       = BB.Bx.values, BB.By.values, BB.Bz.values
      
    #VBangle_mean, dVB, VBangle_std = BVangle(br, bt, bn, vr, vt, vn , smoothed)  

    Va_r = 1e-15* Br/np.sqrt(mu0*df_part['np'].values*m_p)   
    Va_t = 1e-15* Bt/np.sqrt(mu0*df_part['np'].values*m_p)   ### Multuply by 1e-15 to get units of [Km/s]
    Va_n = 1e-15* Bn/np.sqrt(mu0*df_part['np'].values*m_p)   
    
    # Estimate VB angle
    vbang = np.arccos((Va_r * Vr + Va_t * Vt + Va_n * Vn)/np.sqrt((Va_r**2+Va_t**2+Va_n**2)*(Vr**2+Vt**2+Vn**2)))
    vbang = vbang/np.pi*180#
    VBangle_mean, VBangle_std = np.nanmean(vbang), np.nanstd(vbang)
    
    # Also save the components of Vsw an Valfven
    alfv_speed = [np.nanmean(Va_r), np.nanmean(Va_t), np.nanmean(Va_n)]
    sw_speed   = [np.nanmean(Vr), np.nanmean(Vt), np.nanmean(Vn)]

    # sign of Br within the window
    signB = - np.sign(np.nanmean(Va_r))
    
    # # Estimate fluctuations of fields #
    if subtract_rol_mean:
        va_r = Va_r - 1e-15*BB['Br_mean'].values/np.sqrt(mu0*df_part['np_mean'].values*m_p);    v_r = Vr - df_part['Vr_mean'].values
        va_t = Va_t - 1e-15*BB['Bt_mean'].values/np.sqrt(mu0*df_part['np_mean'].values*m_p);    v_t = Vt - df_part['Vt_mean'].values
        va_n = Va_n - 1e-15*BB['Bn_mean'].values/np.sqrt(mu0*df_part['np_mean'].values*m_p);    v_n = Vn - df_part['Vn_mean'].values

        # va_r = Va_r;     v_r  = vr 
        # va_t = Va_t;     v_t  = vt #- df_part['Vt_mean'].values
        # va_n = Va_n;      v_n = vn #- df_part['Vn_mean'].values

    else:
        va_r = Va_r - np.nanmean(Va_r);   v_r = Vr - np.nanmean(Vr)
        va_t = Va_t - np.nanmean(Va_t);   v_t = Vt - np.nanmean(Vt)
        va_n = Va_n - np.nanmean(Va_n);   v_n = Vn - np.nanmean(Vn)
    


    # Estimate Zp, Zm components
    Zpr = v_r +  signB *va_r; Zmr = v_r - signB *va_r
    Zpt = v_t +  signB *va_t; Zmt = v_t - signB *va_t
    Zpn = v_n +  signB *va_n; Zmn = v_n - signB *va_n

    # Zpr = v_r +  va_r; Zmr = v_r - va_r
    # Zpt = v_t +  va_t; Zmt = v_t - va_t
    # Zpn = v_n +  va_n; Zmn = v_n - va_n

    # Estimate energy in Zp, Zm
    Z_plus_squared  = Zpr**2 +  Zpt**2 + Zpn**2
    Z_minus_squared = Zmr**2 +  Zmt**2 + Zmn**2
    
    # Estimate amplitude of fluctuations
    Z_amplitude                = np.sqrt( (Z_plus_squared + Z_minus_squared)/2 ) ; Z_amplitude_mean    = np.nanmedian(Z_amplitude); Z_amplitude_std = np.nanstd(Z_amplitude);

    # Kin, mag energy
    Ek = v_r**2 + v_t**2 + v_n**2
    Eb = va_r**2 + va_t**2 + va_n**2
    
    #Estimate normalized residual energy
    sigma_r      = (Ek-Eb)/(Ek+Eb);                                                         sigma_r[np.abs(sigma_r) > 1e5] = np.nan;
    sigma_c      = (Z_plus_squared - Z_minus_squared)/( Z_plus_squared + Z_minus_squared);  sigma_c[np.abs(sigma_c) > 1e5] = np.nan
    
    #Save in DF format to estimate spectraly
    try:
        distance1 = df_part['Dist_au']
    except:
        distance1 = df_part['d'] 
        
    nn_df       = pd.DataFrame({'DateTime': nindex,
                                'Va_r'    : Va_r,  'Va_t': Va_t,  'Va_n': Va_n,
                                'V_r'     : Vr,    'V_t' : Vt,    'V_n' : Vn, 'di':di,
                                'beta'    : beta,  'np'  : Np,   'Vth'  : Vth, 'd':distance1,
                                'sigma_c' : sigma_c, 'sigma_r': sigma_r, 'vbangle':vbang}).set_index('DateTime')
    nn_df       = nn_df.mask(np.isinf(nn_df)).dropna().interpolate(method='linear')

    
    return  nn_df