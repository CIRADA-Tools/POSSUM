import validation_functions as vf
import sys
import datetime
import params
import pdb

pickleout=sys.argv[1]
incube=sys.argv[2]
filtered_csv=params.filtered_csv
imsize=params.imsize
print("Starting tabulateSpectra for " + incube)
print(datetime.datetime.now())
vf.tabulateSpectra(pickleout, incube, filtered_csv, int(imsize))

		