"""
Initialize the validation: Creates the files necessary to run the validation script
Reads selavy files and create and save dataframes.
Save initialized variables to a pickle file.
Reads the cubes and tabulates the spectra and saves those to a pickle file as well.

"""

#from validation_utils import *
import validation_functions as vf
import validation_utils as vu
import numpy as np
from numpy import inf
import pandas as pd
import sys
import os
import subprocess
import pickle
import pdb
from astropy import units as u
from astropy.coordinates import SkyCoord  # High-level coordinates

import params


path = params.path
outpath = params.outpath
sb = params.sb
base = params.base
account = params.account
runValidation = params.runValidation
doResiduals = params.doResiduals
doWeights = params.doWeights
makeTar = params.makeTar
fluxmin = params.fluxmin #min flux limit in mJy
fluxmax = params.fluxmax #max flux limit in mJy
polmin = params.polmin
polmax = params.polmax
snrmin = params.snrmin
snrmax = params.snrmax
imsize = params.imsize #size of postage stamp image in pixels



def initValidation(path, sb, base, fluxmin=None, fluxmax=None, polmin=None, polmax=None, snrmin=None, snrmax=None, imsize=20, doResiduals=True, doWeights=True, runValidation=True):


	#Define file names
	files=vf.pipelineFiles(path, base, sb)


	## Make directories for outputs:
	vf.createOutputDirs(path)

	#read the mslist file and get some observation parameteres
	#This could fail in the future if the format of the mslist file changes!
	obsVars=vu.getObsVar(files["mslist"])#[ra,dec,freq0,chanWid,ctrFreq,obsStart, obsEnd, obsdate]

	# use the pointing centre defined in the mslist file
	pointing_ra=[obsVars[0]]
	pointing_dec=[obsVars[1]]
	centre_coord = SkyCoord(pointing_ra[0]+" "+pointing_dec[0].replace('.',':',2), frame='icrs',unit=[u.h,u.deg]) 
	centre_coord_gal = centre_coord.transform_to('galactic')

	#merge the selavy-component and selavy-polarization files together into a pandas dataframe
	merged=vf.mergeSelavy(files['selavy_comp'], files['selavy_pol'])
	tot_numsources=merged.shape[0]
	############## add some columns to the merged selavy file ##############
	############## these are columns to appear on the webpage ##############


	#get distaces from beam and mosaic centres
	mosdistances,mos_pa = vu.getMosaicDistances(centre_coord.ra.deg, centre_coord.dec.deg, merged["ra_deg_cont"].values, merged["dec_deg_cont"].values)
	beam_dist,beam_pa,beam_num = vu.getBeamDistances(files['footprint'], merged["ra_deg_cont"].values, merged["dec_deg_cont"].values)

	#insert these columns into the merged dataframe (into the 8th column - position doesn't really matter at this point)
	merged.insert(8,"PA in beam", beam_pa)
	merged.insert(8,"Dist to beam", beam_dist)
	merged.insert(8,"Nearest beam", beam_num)
	merged.insert(8,"Dist to centre", mosdistances)

	#filter the merged dataframe
	filtered=vf.filterDataframe(merged, fluxmin=fluxmin, fluxmax=fluxmax)#, polmin=None, polmax=None, snrmin=None, snrmax=None)
	print("Running validation on " +str(len(filtered["component_name"])) +" sources.")
	#add columns that include the hover links and links to the FITS cubelets
	plotfilename="plots/sources/" + filtered["component_name"].astype(str) + ".png"
	linktext=filtered["component_name"].astype(str)
	filtered["source"] = vf.makeHoverLink(plotfilename, linktext)

	filename="polIm/image.restored.i.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
	linktext = "FITS (im)"
	filtered["ifits"] = vf.makeFileLink(filename, linktext)
	filename="polIm/image.restored.q.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
	filtered["qfits"] = vf.makeFileLink(filename, linktext)
	filename="polIm/image.restored.u.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
	filtered["ufits"] = vf.makeFileLink(filename, linktext)
	filename="polIm/image.restored.v.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
	filtered["vfits"] = vf.makeFileLink(filename, linktext)
	
	if(doResiduals):
		plotfilename="plots/sources/residuals." + filtered["component_name"].astype(str) + ".png"
		linktext="RESIDUAL"
		filtered["residuals"] = vf.makeHoverLink(plotfilename, linktext)
		filename="polIm/residuals.i.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		linktext = "FITS (res)"
		filtered["i_res_fits"] = vf.makeFileLink(filename, linktext)
		filename="polIm/residuals.q.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		filtered["q_res_fits"] = vf.makeFileLink(filename, linktext)
		filename="polIm/residuals.u.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		filtered["u_res_fits"] = vf.makeFileLink(filename, linktext)
		filename="polIm/residuals.v.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		filtered["v_res_fits"] = vf.makeFileLink(filename, linktext)

	if(doWeights):
		plotfilename="plots/sources/weights." + filtered["component_name"].astype(str) + ".png"
		linktext="WEIGHT"
		filtered["weights"] = vf.makeHoverLink(plotfilename, linktext)
		filename="polIm/weights.i.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		linktext = "FITS (res)"
		filtered["i_weight_fits"] = vf.makeFileLink(filename, linktext)
		filename="polIm/weights.q.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		filtered["q_weight_fits"] = vf.makeFileLink(filename, linktext)
		filename="polIm/weights.u.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		filtered["u_weight_fits"] = vf.makeFileLink(filename, linktext)
		filename="polIm/weights.v.SB"+str(sb)+"."+filtered["component_name"].astype(str)+".fits"
		filtered["v_weight_fits"] = vf.makeFileLink(filename, linktext)

	#write the merged dataframe and filtered dataframe to files for reference

	merged.to_csv(path+"validation/merged.csv")
	filtered_csv=path+"validation/filtered.csv"
	filtered.to_csv(filtered_csv)

	incube=[]
	pickleout=[]
	ra=[]
	dec=[]

	for i in range(12):
		#if(verified[i]): #check if file exists
		key=list(files.keys())[i]
		value=list(files.values())[i]
		os.system("mkdir "+ path+key)
		#os.system("mv "+ path+value + " " +path+key )
		os.system("ln -s "+path+value+" "+path+key+"/"+value)
		value=path+key+"/"+value
		files[key] = value
		incube.append(value)
		os.system("mkdir "+ path+"validation/"+key)
		pickleout.append(path+"validation/"+key+"/"+key+".pickle")
		ra.append(filtered["ra_deg_cont"].values)
		dec.append(filtered["dec_deg_cont"].values)

	#sbatch file parameters 
	sbatch=path+"validation/tabulateSpectra.sbatch"
	account = account
	time="8:00:00"
	nodes="1"
	ntasks=str(len(incube)+1)
	#slurmout="/scratch/b/bmg/westjl/slurmout/slurm-tabulateSpectra-%j.out"
	slurmout=outpath+"/possum-validation-"+sb+"-%j.out"
	
	#create the sbatch script that tabulates the spectrum and saves them to pickle files
	vf.make_galaxy_sbatch(sbatch, account, time, nodes, ntasks, sb, slurmout, pickleout, incube, filtered_csv, imsize, runValidation, makeTar, path)
	#run the sbatch script
	os.system('sbatch '+sbatch)

	#Verify that the files exist
	verified=vf.verifyFiles(files)

	
	picklefile=path+"validation/variables.pickle"
	variables= {'path':path, 'sb': sb, 'base' : base, 'tot_numsources': tot_numsources, 'fluxmin' : fluxmin, 'fluxmax' : fluxmax,\
				'polmin':polmin, 'polmax':polmax, 'snrmin':snrmin, 'snrmax':snrmax, 'imsize':imsize,\
				'files':files, 'verified':verified, 'pickleout':pickleout, 'obsVars':obsVars,\
				'centre_coord':centre_coord, 'centre_coord_gal':centre_coord_gal}
	   
	with open(picklefile, 'wb') as fp:
		pickle.dump(variables, fp)


initValidation(path, sb, base, fluxmin=fluxmin, fluxmax=fluxmax, polmin=polmin, polmax=polmax, snrmin=snrmin, snrmax=snrmax, imsize=imsize, doResiduals=doResiduals, doWeights=doWeights, runValidation=runValidation)
