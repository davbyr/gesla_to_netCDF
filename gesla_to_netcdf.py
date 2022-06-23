'''
v1.0

This script is for reading multiple  GESLA files from a directory and
putting the data onto a regular single time format, ready for a COAsT Tidegauge
object. You can specify the dates you want to interpolate onto and the frequency.
Time interpolation is a nearest neighbour interpolation. Any nearest points that
are too far from the original data will be turned to NaNs. You can specify
how far  you are happy for the interpolation to work using the max_diff 
variable. By default this is set to 5 minutes for hourly output.
Output will be saved to file if save_output is True

You can optionally do an additional harmonic analysis using COAsT and utide.
To do this set do_harmonic_analysis to True. fn_analysis is where the 
harmonics will be saved.

NOTE: By default, the harmonic analysis is performed using utide's application
of the Rayleigh criterion. It may be that not all requested constituents are
available if the time series is NaN. In this case, NaNs will be returned in the
output file.
'''

# Import libraries
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
#sys.path.append("<Development path to COAST>")
import coast
from datetime import datetime
import pandas as pd
import scipy.interpolate as interp

# Directory to GESLA files
fn_tg = '/Users/dbyrne/Projects/SEASia/gesla/*'
save_output = True
fn_out = '/Users/dbyrne/Projects/SEASia/gesla/tg_coast.nc'

# Settings - output time array, start, end and frequency
start_date = datetime(1986,1,1)
end_date = datetime(2023,1,1)
freq_out = '1H'

# Do you want the time series demeaned?
demean_ssh = True

# Settings - These are the quality control flags that will be kept
# 0 - no QC
# 1 - correct value
# 2 - interpolated value
# 3 - doubtful value
# 4 - isolated spike/wrong value
# 5 - missing value
qc_flags_to_keep = [1, 2]

# Maximum time difference within which to keep nearest neighbours (keep small)
max_diff = 5 #minutes

# Harmonic analysis settings
do_harmonic_analysis = True
fn_analysis = '/Users/dbyrne/Projects/SEASia/gesla/tg_analysis.nc'
constituents_to_save = ['M2','S2','N2','K1','O1','P1']


###############################################################################
###############################################################################

# Read in gesla files
Tidegauge = coast.Tidegauge()
tg_list = Tidegauge.read_gesla_v3(fn_tg)

# Get number of tidegauges
n_tg = len(tg_list)

# Loop over tidegauges in list and apply quality control flags (set to NaN where bad)
ii = 0
for tg in tg_list:
    qcf = tg.dataset.qc_flags[0].values
    ind_to_keep = np.zeros(qcf.shape)
    for qq in qc_flags_to_keep:
        check_array = qcf == qq
        ind_to_keep = np.logical_or(check_array, ind_to_keep)
        
    tg.dataset['ssh'] = tg.dataset.ssh.isel(id_dim=0).where(ind_to_keep).expand_dims('id_dim')
    
# Regularize time
date_out = pd.date_range(start_date, end_date, freq=freq_out)

# Do nearest neighbour interpolation
ssh_out = np.zeros((n_tg, len(date_out)))*np.nan
lon_out = np.zeros(n_tg)
lat_out = np.zeros(n_tg)
name_out = []
for tg_ii in range(n_tg):
    tg = tg_list[tg_ii]
    ssh = tg.dataset.ssh.values[0]
    time = pd.to_datetime( tg.dataset.time.values )
    
    # Get nearest indices
    indices = np.arange(0, len(time))
    f = interp.interp1d(np.array(time).astype('float'), indices, 
                        kind='nearest', fill_value='extrapolate')
    interp_indices = f(np.array(date_out).astype('float')).astype('int')
    
    # Get differences
    time_diff = np.abs(date_out - time[interp_indices]) 
    time_diff_mins = time_diff.total_seconds()/60
    
    # Get all points that are close enough
    close_enough = time_diff_mins <= max_diff
    
    # Finally, index the ssh array to get interpolated values
    ssh_interp = ssh[interp_indices]
    ssh_interp[~close_enough] = np.nan
    
    if demean_ssh:
        ssh_interp = ssh_interp - np.nanmean(ssh_interp)
    
    # Add to output array
    ssh_out[tg_ii] = ssh_interp
        
    
    # Populate lons, lats and names
    lon_out[tg_ii] = tg.dataset.longitude.values
    lat_out[tg_ii] = tg.dataset.latitude.values
    name_out.append( tg.dataset.id_name.values[0] )
    
    
# Create output dataset
ds_out = xr.Dataset( coords = dict(
                         longitude = (['id_dim'], lon_out),
                         latitude = (['id_dim'], lat_out),
                         id_name = (['id_dim'], name_out),
                         time = (['t_dim'], date_out)),
                     data_vars = dict(
                         ssh = (['id_dim','t_dim'], ssh_out)))

# Save to output file
if save_output:
    ds_out.to_netcdf(fn_out)

if do_harmonic_analysis:
    # Create analysis object and get numbers of tidegauges and constituents
    tga = coast.TidegaugeAnalysis() 
    n_tg = ds_out.dims['id_dim']
    n_const = len(constituents_to_save)
    
    # Create output arrays
    amplitude = np.zeros((n_tg, n_const))*np.nan
    phase = np.zeros((n_tg, n_const))
    
    # Loop over tide gauges and do harmonic analysis
    harmonics = tga.harmonic_analysis_utide(ds_out.ssh)
    
    # Loop over tidegauges and extract harmonics
    for tii in range(n_tg):
        ha_tii = harmonics[tii]
        
        # Loop over constituents
        for cii in range(n_const):
            cc = constituents_to_save[cii]
            
            # Check if constituent is available in output
            if cc in ha_tii.name:
                cc_index = np.where( ha_tii.name == cc )[0][0]
                amplitude[tii, cii] = ha_tii.A[cc_index]
                phase[tii, cii] = ha_tii.g[cc_index]
                
    # Create output dataset
    ds_ha = xr.Dataset( coords = dict(
                            longitude = (['id_dim'], lon_out),
                            latitude = (['id_dim'], lat_out),
                            id_name = (['id_dim'], name_out),
                            constituent = (['constituent'], constituents_to_save)),
                        data_vars = dict(
                            amplitude = (['id_dim','constituent'], amplitude),
                            phase = (['id_dim','constituent'], phase)))
    
    # Save to output
    ds_ha.to_netcdf(fn_analysis)
