"""
Make all the validation images and the html file to display them.
"""

#from validation_utils import *
import validation_functions as vf
import validation_utils as vu

import numpy as np
from numpy import inf
import re
import pandas as pd
import astropy.io.fits as pf
from astropy.wcs import WCS
import astropy as ast

import sys
import os
import pickle
import datetime
import pdb

import params


#python doValidation.py /home/b/bmg/westjl/jlwscratch/pilot/10168/validation/variables.pickle /home/b/bmg/westjl/jlwscratch/pilot/10168/validation/filtered.csv True
variables=params.variables
filtered_csv=params.filtered_csv
merged_csv=params.merged_csv
makePlotsForSources=params.makePlotsForSources
doResiduals=params.doResiduals
doWeights=params.doWeights
makeTar=params.makeTar

#variables=path+"validation/variables.pickle"

def doValidation(var, filtered_csv, merged_csv, makePlotsForSources, doResiduals=True, doWeights=True):

    print("Starting doValidation")
    print(datetime.datetime.now())
    #read the pandas dataframe
    filtered = pd.read_csv(filtered_csv)
    merged = pd.read_csv(merged_csv)

    #read the pickle files
    with open (var, 'rb') as fp:
            variables = pickle.load(fp)

    pickleout = variables['pickleout']
    files = variables['files']
    obsVars = variables['obsVars']
    verified = variables['verified']
    sb = variables['sb']
    base = variables['base']
    path =variables['path']
    imsize=variables['imsize']
    fluxmin=variables['fluxmin']
    fluxmax=variables['fluxmax']
    tot_numsources = variables['tot_numsources']
    centre_coord = variables['centre_coord']
    centre_coord_gal = variables['centre_coord_gal']
    
    if(verified[0]):
        with open (pickleout[0], 'rb') as fp:
            iStats = pickle.load(fp)
    if(verified[1]):
        with open (pickleout[1], 'rb') as fp:
            qStats = pickle.load(fp)
    if(verified[2]):
        with open (pickleout[2], 'rb') as fp:
            uStats = pickle.load(fp)
    if(verified[3]):
        with open (pickleout[3], 'rb') as fp:
            vStats = pickle.load(fp)

    if(doResiduals):
        if(verified[4]):
            with open (pickleout[4], 'rb') as fp:
                i_resStats = pickle.load(fp)
        if(verified[5]):
            with open (pickleout[5], 'rb') as fp:
                q_resStats = pickle.load(fp)
        if(verified[6]):
            with open (pickleout[6], 'rb') as fp:
                u_resStats = pickle.load(fp)
        if(verified[7]):
            with open (pickleout[7], 'rb') as fp:
                v_resStats = pickle.load(fp)
    if(doWeights):
        if(verified[8]):
            with open (pickleout[8], 'rb') as fp:
                i_wtStats = pickle.load(fp)
        if(verified[9]):
            with open (pickleout[9], 'rb') as fp:
                q_wtStats = pickle.load(fp)
        if(verified[10]):
            with open (pickleout[10], 'rb') as fp:
                u_wtStats = pickle.load(fp)
        if(verified[11]):
            with open (pickleout[11], 'rb') as fp:
                v_wtStats = pickle.load(fp)
    
    #Get Taylor term image data
    t0_res = pf.getdata(files["iRes_t0"])[:,0,:,:]
    fieldrms=np.nanstd(t0_res) #Jy/beam
    numSources = len(filtered["source"])
    head = pf.getheader(files["iCube"])
    #make individual source plots
    if(makePlotsForSources):
        vu.makePlotsForSources(filtered, iStats, qStats, uStats, vStats, files["i_t0"], files["i_t1"], path, sb, head, imsize)
        if(doResiduals):
            vu.makePlotsForResiduals(filtered, i_resStats, q_resStats, u_resStats, v_resStats, files["iRes_t0"], "residuals", path, sb, head, imsize)
        if(doWeights):
            vu.makePlotsForResiduals(filtered, i_wtStats, q_wtStats, u_wtStats, v_wtStats, files["iWt_t0"], "weights", path, sb, head, imsize)
    

###################### Make plots for webpage ###############################
##### Append links to image_links=[] 
##### and then pass them all to be writen to the html file
##### plotfilename links are relative links to work on the webpage
#############################################################################   
    
    image_links=[]

    title = str(numSources)+" sources with Taylor 0 flux from "+"{0:.1g}".format(fluxmin/1000)+" to "+"{0:.1g}".format(fluxmax/1000)+" Jy"
    i_vals_cat = np.concatenate(iStats['Vals'])
    v_vals_cat = np.concatenate(vStats['Vals'])
    q_vals_cat = np.concatenate(qStats['Vals'])
    u_vals_cat = np.concatenate(uStats['Vals'])

    plotfilename="plots/StokesI_histogram.png"
    ihiststats=vu.makeHistogram(i_vals_cat, "Stokes I", title, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes I histogram"))
    
    plotfilename="plots/StokesV_histogram.png"
    vhiststats=vu.makeHistogram(v_vals_cat, "Stokes V", title, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes V histogram"))
    
    plotfilename="plots/StokesQ_histogram.png"
    qhiststats=vu.makeHistogram(q_vals_cat, "Stokes Q", title, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes Q histogram"))
    
    plotfilename="plots/StokesU_histogram.png"
    uhiststats=vu.makeHistogram(u_vals_cat, "Stokes U", title, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes U histogram"))
        
    plotfilename="plots/Leakage_histogram.png"
    leakagehiststats=vu.makeHistogram(filtered["med_abs_leakage"], "Leakage", title, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Leakage histogram"))
    
    plotfilename="plots/FaradayDepth_histogram.png"
    fdstats=vu.makeHistogram(filtered['fd_peak_fit'].values, "Faraday Depth", title, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Faraday Depth histogram"))

    plotfilename="plots/stokesV_vs_dist_from_mosaic_centre.png"
    vu.makeVvsDist(filtered["Dist to centre"].values, vStats['Sixteenth'], vStats['EightyForth'], vStats['Medians'], vStats['StDev'], "Distance from mosaic centre (degrees)", fluxmin, fluxmax, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes V vs dist from mosaic centre"))
    
    plotfilename="plots/stokesV_vs_dist_from_beam_centre.png"
    vu.makeVvsDist(filtered["Dist to beam"].values, vStats['Sixteenth'], vStats['EightyForth'], vStats['Medians'], vStats['StDev'],"Distance from beam centre (degrees)", fluxmin, fluxmax, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes V vs dist from beam centre"))
    
    plotfilename="plots/stokesVoverI_vs_dist_from_mosaic_centre.png"
    vu.makeVvsDist(filtered["Dist to centre"].values, vStats['Sixteenth']/iStats['Sixteenth'], vStats['EightyForth']/iStats['EightyForth'], vStats['Medians']/iStats['Medians'], vStats['StDev']/iStats['StDev'], "V/I vs Distance from mosaic centre (degrees)", fluxmin, fluxmax, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes V/I vs dist from mosaic centre"))
    
    plotfilename="plots/stokesV_vs_dist_from_beam_centre.png"
    vu.makeVvsDist(filtered["Dist to beam"].values, vStats['Sixteenth']/iStats['Sixteenth'], vStats['EightyForth']/iStats['EightyForth'], vStats['Medians']/iStats['Medians'], vStats['StDev']/iStats['StDev'],"V/I vs Distance from beam centre (degrees)", fluxmin, fluxmax, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Stokes V/I vs dist from beam centre"))

    plotfilename="plots/leakage.png"
    vu.makeLeakagePlot(filtered["PA in beam"].values, filtered["Nearest beam"].values, filtered["Dist to beam"].values, filtered["Dist to beam"].values, path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Leakage Map"))
    
    plotfilename="plots/logNlogP.png"
    vu.makeLogNLogP(filtered[(filtered.pol_peak_fit_snr < 1000)], 30, 36, fluxmin, fluxmax, path+plotfilename) #[(filtered.pol_peak_fit_snr < 1000)]
    image_links.append(vf.makeHoverLink(plotfilename,"Log N vs Log P"))

    plotfilename="plots/polvsStokesI.png"
    vu.makeScatter(filtered['flux_I_median (mJy/beam)'].values, filtered['pol_peak'].values, filtered['rms_I'].values, filtered['pol_peak_err'].values, 0.0, 3*np.nanstd(filtered['flux_I_median (mJy/beam)'].values), 0.0, 3*np.nanstd( filtered[(filtered.pol_peak < 1000)]["pol_peak"].values ), "Median Stokes I Flux (mJy)", "Peak pol (mJy)", "Polarized flux vs Total Flux", path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Peak Pol vs Stokes I"))

    plotfilename="plots/FDvsPolfrac.png"
    vu.makeScatter(filtered['fd_peak_fit'].values, filtered['pol_peak/flux_I_med'].values, filtered["fd_peak_fit_err"].values, np.zeros(filtered['pol_peak_err'].values.shape), -150.0, 150.0, 0.0, 0.5, "Fitted FD peak (rad/m^2)", "Polarized Fraction", "Polarized fraction vs FD Peak", path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Polarized fraction vs FD Peak"))

    plotfilename="plots/StokesInoisevsfreq.png"
    vu.makeScatter(iStats['Freq'][0], np.nanmean(iStats['EdgePix'],axis=0), np.zeros(iStats['Freq'][0].shape), np.nanstd(iStats['EdgePix'],axis=0), np.nanmin(np.array(iStats['Freq'][0])), np.nanmax(np.array(iStats['Freq'][0])),-2.5*np.nanstd(iStats['EdgePix']), 2.5*np.nanstd(iStats['EdgePix']), "Freq (Hz)", "Stokes I Uncertainty", "Average Stokes I Uncertainty vs Freq ("+str(numSources)+" sources)", path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Average Stokes I Noise vs Freq"))

    plotfilename="plots/StokesVnoisevsfreq.png"
    vu.makeScatter(vStats['Freq'][0], np.nanmean(vStats['EdgePix'],axis=0), np.zeros(vStats['Freq'][0].shape), np.nanstd(vStats['EdgePix'],axis=0), np.nanmin(np.array(vStats['Freq'][0])), np.nanmax(np.array(vStats['Freq'][0])), -2.5*np.nanstd(vStats['EdgePix']), 2.5*np.nanstd(vStats['EdgePix']), "Freq (Hz)", "Stokes V Uncertainty", "Average Stokes V Uncertainty vs Freq ("+str(numSources)+" sources)", path+plotfilename)
    image_links.append(vf.makeHoverLink(plotfilename,"Average Stokes V Noise vs Freq"))

    if(doResiduals):
        title = "Resiuals for "+str(numSources)+" sources with Taylor 0 flux from "+"{0:.1g}".format(fluxmin/1000)+" to "+"{0:.1g}".format(fluxmax/1000)+" Jy"

        ires_vals_cat = np.concatenate(i_resStats["Vals"])
        vres_vals_cat = np.concatenate(v_resStats["Vals"])
        qres_vals_cat = np.concatenate(q_resStats["Vals"])
        ures_vals_cat = np.concatenate(u_resStats["Vals"])

        plotfilename="plots/StokesI_residuals_histogram.png"
        ireshiststats=vu.makeHistogram(ires_vals_cat, "Stokes I Residuals", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes I histogram - Residuals"))

        plotfilename="plots/StokesV_residuals_histogram.png"
        vreshiststats=vu.makeHistogram(vres_vals_cat, "Stokes V Residuals", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes V histogram - Residuals"))
        
        plotfilename="plots/StokesQ_residuals_histogram.png"
        qreshiststats=vu.makeHistogram(qres_vals_cat, "Stokes Q Residuals", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes Q histogram - Residuals"))

        plotfilename="plots/StokesU_residuals_histogram.png"
        ureshiststats=vu.makeHistogram(ures_vals_cat, "Stokes U Residuals", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes U histogram - Residuals"))
        
        plotfilename="plots/StokesInoiseResidualsvsfreq.png"
        vu.makeScatter(i_resStats['Freq'][0], np.nanmean(i_resStats['EdgePix'],axis=0), np.zeros(i_resStats['Freq'][0].shape), np.nanstd(i_resStats['EdgePix'],axis=0), np.nanmin(np.array(i_resStats['Freq'][0])), np.nanmax(np.array(i_resStats['Freq'][0])),-2.5*np.nanstd(i_resStats['EdgePix']), 2.5*np.nanstd(i_resStats['EdgePix']), "Freq (Hz)", "Stokes I Resiuals Uncertainty (off source)", "Average Stokes I Resiuals Uncertainty (off source) vs Freq ("+str(numSources)+" sources)", path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Average Stokes I Resiuals Uncertainty (off source) vs Freq"))


        plotfilename="plots/StokesVnoiseResidualsvsfreq.png"
        vu.makeScatter(v_resStats['Freq'][0], np.nanmean(v_resStats['EdgePix'],axis=0), np.zeros(v_resStats['Freq'][0].shape), np.nanstd(v_resStats['EdgePix'],axis=0), np.nanmin(np.array(v_resStats['Freq'][0])), np.nanmax(np.array(v_resStats['Freq'][0])), -2.5*np.nanstd(v_resStats['EdgePix']), 2.5*np.nanstd(v_resStats['EdgePix']), "Freq (Hz)", "Stokes V Resiuals Uncertainty (off source)", "Average Stokes V Resiuals Uncertainty (off source) vs Freq ("+str(numSources)+" sources)", path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Average Stokes V Resiuals Uncertainty (off source) vs Freq"))


    if(doWeights):
        title = "Weights for "+str(numSources)+" sources with Taylor 0 flux from "+"{0:.1g}".format(fluxmin/1000)+" to "+"{0:.1g}".format(fluxmax/1000)+" Jy"
        iw_vals_cat = np.concatenate(np.array(i_wtStats["Vals"]))
        vw_vals_cat = np.concatenate(np.array(v_wtStats["Vals"]))
        qw_vals_cat = np.concatenate(np.array(q_wtStats["Vals"]))
        uw_vals_cat = np.concatenate(np.array(u_wtStats["Vals"]))

        plotfilename="plots/StokesI_weights_histogram.png"
        iwhiststats=vu.makeHistogram(iw_vals_cat, "Stokes I Weights", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes I histogram - Weights"))
        
        plotfilename="plots/StokesV_weights_histogram.png"
        vwhiststats=vu.makeHistogram(vw_vals_cat, "Stokes V Weights", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes V histogram - Weights"))

        plotfilename="plots/StokesQ_weights_histogram.png"
        qwhiststats=vu.makeHistogram(qw_vals_cat, "Stokes Q Weights", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes Q histogram - Weights"))

        plotfilename="plots/StokesU_weights_histogram.png"
        uwhiststats=vu.makeHistogram(uw_vals_cat, "Stokes U Weights", title, path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Stokes U histogram - Weights"))

        plotfilename="plots/StokesInoiseWeightsvsfreq.png"
        vu.makeScatter(i_wtStats['Freq'][0], np.nanmean(i_wtStats['EdgePix'],axis=0), np.zeros(i_wtStats['Freq'][0].shape), np.nanstd(i_wtStats['EdgePix'],axis=0), np.nanmin(np.array(i_wtStats['Freq'][0])), np.nanmax(np.array(i_wtStats['Freq'][0])),-2.5*np.nanstd(i_wtStats['EdgePix']), 2.5*np.nanstd(i_wtStats['EdgePix']), "Freq (Hz)", "Stokes I Resiuals Uncertainty (off source)", "Average Stokes I Resiuals Uncertainty (off source) vs Freq ("+str(numSources)+" sources)", path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Average Stokes I Weights Uncertainty (off source) vs Freq"))
        
        plotfilename="plots/StokesVnoiseWeightsvsfreq.png"
        vu.makeScatter(v_wtStats['Freq'][0], np.nanmean(v_wtStats['EdgePix'],axis=0), np.zeros(v_wtStats['Freq'][0].shape), np.nanstd(v_wtStats['EdgePix'],axis=0), np.nanmin(np.array(v_wtStats['Freq'][0])), np.nanmax(np.array(v_wtStats['Freq'][0])), -2.5*np.nanstd(v_wtStats['EdgePix']), 2.5*np.nanstd(v_wtStats['EdgePix']), "Freq (Hz)", "Stokes V Resiuals Uncertainty (off source)", "Average Stokes V Resiuals Uncertainty (off source) vs Freq ("+str(numSources)+" sources)", path+plotfilename)
        image_links.append(vf.makeHoverLink(plotfilename,"Average Stokes V Weights Uncertainty (off source) vs Freq"))

###################### Make tables for webpage ###############################
##### Create tables with observation and source statistics
##### Can add extra rows to these tables 
#############################################################################   

    
    
    beamCoords=vu.hms2skycoord(vu.read_centers(files['footprint']))
    solarfluxmean, solarfluxstd, jd = vu.getSolarFlux([obsVars[7]])
    Op_fd, Op_fd_err = vu.getOppermannAv(files["OpData"], files["OpErr"], beamCoords)
    sun_coord=ast.coordinates.get_sun(ast.time.Time(jd, format='jd', scale='utc'))
    sun_dist=sun_coord.separation(centre_coord).value
  

    obsStats = pd.DataFrame()
    obsStats["Observation start time"] = [obsVars[5]]
    obsStats["Observation end time"] = [obsVars[6]]
    obsStats["RA (pointing centre)"] = [re.split('\s+', centre_coord.to_string('hmsdms'))[0]]
    obsStats["Dec (pointing centre)"] = [centre_coord.dec.deg]
    obsStats["Gal. lon (pointing centre)"] = [centre_coord_gal.l.deg]
    obsStats["Gal. lat (pointing centre)"] = [centre_coord_gal.b.deg]
    obsStats["RMS of Field (microJy/beam)"] = ['{:20,.2f}'.format(fieldrms*1000000)]
    obsStats["Freq. of Chan 0 (MHz)"] = [obsVars[2]]
    obsStats["Chan Width (kHz)"] = [obsVars[3]]
    obsStats["Centre Freq. (MHz)"] = [obsVars[4]]
    obsStats["Solar Flux (sfu) for Obs Date"] = ['{:20,.2f}'.format(solarfluxmean)+" +/- "+'{:20,.2f}'.format(solarfluxstd)]
    obsStats["Dist to the Sun (deg)"] = ['{:20,.2f}'.format(sun_dist)]
    oST=obsStats.transpose()

    fieldStats = pd.DataFrame()
    
    fieldStats["Number of components"] = [str(int(tot_numsources)), str(int(numSources))]
    fieldStats["Average flux - Stokes I (mJy/beam)"] = [np.mean(merged["flux_I_median"].values), np.mean(filtered["flux_I_median (mJy/beam)"].values)]
    fieldStats["Stdev - Stokes I (mJy/beam)"] = [np.std(merged["flux_I_median"].values), np.std(filtered["flux_I_median (mJy/beam)"].values)]
    fieldStats["Average flux - Stokes V (mJy/beam)"] = [np.mean(merged["flux_V_median"].values), np.mean(filtered["flux_V_median"].values)]
    fieldStats["Stdev - Stokes V (mJy/beam)"] = [np.std(merged["flux_V_median"].values), np.std(filtered["flux_V_median"].values)]
    fieldStats["# of components with Stokes I < 0"] = [sum(merged["flux_I_median"].values<0), sum(filtered["flux_I_median (mJy/beam)"].values<0)]
    fieldStats["# of components with bad max PI"] = [sum(merged["pol_peak"].values>10000), sum(filtered["pol_peak"].values>10000)]
    fieldStats["Average Pol. Frac."] = [np.mean(merged["pol_peak/flux_I_med"].values), np.mean(filtered["pol_peak/flux_I_med"].values)]
    fieldStats["Median Pol. Frac."] = [np.median(merged["pol_peak/flux_I_med"].values), np.median(filtered["pol_peak/flux_I_med"].values)]
    fieldStats["Average FD"] = [np.mean(merged["fd_peak_fit"].values), np.mean(filtered["fd_peak_fit"].values)]
    fieldStats["Median FD"] = [np.median(merged["fd_peak_fit"].values), np.median(filtered["fd_peak_fit"].values)]
    fieldStats["Stdev FD"] = [np.std(merged["fd_peak_fit"].values), np.std(filtered["fd_peak_fit"].values) ]
    fieldStats["Oppermann Map FD with uncertainty"] = ['{:20,.2f}'.format(Op_fd)+"+/-"+'{:20,.2f}'.format(Op_fd_err), "" ]
    fsT=fieldStats.transpose()
    fsT.columns = ["All Stokes I components", "Components: "+"{0:.1f}".format(fluxmin)+" to "+"{0:.1g}".format(fluxmax)+" mJy"]


###############Select which columns to appear on the webpage
######These will appear in the order below
######These can be reordered as desired.
#############################################################################   

    web_dataframe=filtered[['component_id',\
                       "ra_deg_cont",\
                       "dec_deg_cont",\
                       "source",\
                       "residuals",\
                       "weights",\
                       "flux_peak (mJy/beam)",
                       "flux_I_median (mJy/beam)",\
                       "rms_I",\
                       "Dist to centre",\
                       "Nearest beam",\
                       "Dist to beam",\
                       "PA in beam",\
                       "med_abs_leakage",\
                       "ifits", \
                       "qfits", \
                       "ufits", \
                       "vfits", \
                       "i_res_fits", \
                       "q_res_fits", \
                       "u_res_fits", \
                       "v_res_fits", \
                       "i_weight_fits", \
                       "q_weight_fits", \
                       "u_weight_fits", \
                       "v_weight_fits", \
                       "pol_peak",\
                       "pol_peak_err",\
                       'pol_peak_fit_snr',\
                       "pol_peak/flux_I_med",\
                       "fd_peak_fit",\
                       "fd_peak_fit_err",\
                       "pol_ang_ref", \
                       "pol_ang_ref_err", \
                       "pol_ang_zero", \
                       "pol_ang_zero_err",\
                       "lambda_ref_sq",\
                       "rmsf_fwhm",\
                       "flux_Q_median",\
                       "rms_Q",\
                       "flux_U_median",\
                       "rms_U",\
                       "flux_V_median",\
                       "rms_V",\
                       "spectral_index"]]
#### Make the CSS file 
####(controls the hover behaviour and must be included with the html file)
    vf.writeCSS(path)
#### Make the HTML file 
    vf.makeHTML(path, sb, base, fluxmin, fluxmax, fsT, oST, web_dataframe, doResiduals, doWeights, image_links, makeTar)

doValidation(variables, filtered_csv, merged_csv, makePlotsForSources, doResiduals, doWeights)

