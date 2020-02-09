# POSSUM
Data analysis and validation of the POSSUM pipeline

These are scripts to run validation on ASKAP data. It uses pipeline outputs from the ASKAP pipeline.

Basic parameters (such as the SBID and root paths) for running the scripts are specified in the file params.py

This file must be edited (or generated) before running the validation script.

Submitting the main script (runValidation.sh) will initiate the validation. Depending on how the parameters in params.py are set, this will run all or some of the following steps:

- initialization (initValidation.py): reads the selavy files and then creates and submits an sbatch script (tabulateSpectra.sbatch)
- tabulateSpectra: extracts the data for selected sources from the cubes and saves these to python pickle files, then runs doValidation.py
- doValidation: reads the picklefiles created in the previous steps and from these, it makes all the fits cubelets and png files (source plots) and summary plots, tabulates summary statistics and creates the output HTML

These steps can be run individually. In particular, doValidation can be run at a later time, or can be run multiple times, to alter the plot settings or alter the html output.

The scripts require the following files:

- restored image cubes (Stokes IQUV)
- taylor0 and taylor1 restored images (Stokes I)
- optional: residual image cubes (IQUV)
- Stokes I taylor0 residual image (for computing field rms)
- optional: weight image cubes (IQUV)

- selavy outputs for Stokes I components and polarization (within its parent directory)

- footprint fille (in the metadata/ directory)
- mslist file (in the metadata/ directory) (mslist only for the observation, not the cal)

The scripts assume the file structure and naming convention from the ASKAP pipeline. It constructs the file names given the SBID (sb) and "base" file name (as specified in params.py).

It  assumes the following naming convention:

e.g. image.restored.i.SB4612.contcube.fits

-> "image.restored.i."+base+"SB"+str(sb)+".contcube.fits" 


The script will make a directory for each image cube and puts a symbolic link to the files in each directory. It does this because writing to the same directory from different processes can lead to slow down because of directory locking. (But it's not clear if this is actually working?)

