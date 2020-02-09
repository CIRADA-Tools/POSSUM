
#python initValidation.py /home/b/bmg/westjl/jlwscratch/pilot/10168/ 10168 "" 500.0 10000.0

#ASKAP specific variables
sb='10168'
base=""

#root path to pipeline files (image cubes and source finding output)
path="/home/b/bmg/westjl/jlwscratch/pilot/10168/"

#location of output files
variables=path+"validation/variables.pickle"
filtered_csv=path+"validation/filtered.csv"
merged_csv=path+"validation/merged.csv"

#cutoffs for filtering source list
fluxmin=2.0    #mJy/beam
fluxmax=100000.0 #mJy/beam
polmin=None
polmax=None
snrmin=None
snrmax=None

#size of postage stamp image (and cubelet FITS files) in pixels 
imsize=20  

#make the individual plots for sources (i.e., png files)
makePlotsForSources=True

#make plots for the residual cube?
doResiduals=True

#make plots for the weight cube?
doWeights=True

#make a tar archive of the FITS files?
makeTar = True

#run full validation to the end 
# else just creates summary csv files (from selavy) and tabulates the spectra and saves
# to pickle files. These can be used to run the validation separately at a later time
runValidation=True