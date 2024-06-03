#!/usr/bin/env python

from astropy.io import ascii
from astroquery.utils.tap.core import Tap
from astroquery.casda import Casda
import numpy as np
import sys
import os
import getpass
import argparse
import re

def download(url_list, sb):
    
    for url in url_list:
        if url.find("checksum") != -1:
            continue
        print("Processing %s" %(url))
        #print("wget --content-disposition \"%s\"" %(url))
        os.system("wget -P "+sb+" --content-disposition \"%s\"" %(url))
        #os.system("wget --content-disposition \"%s\" -O CMPL.tar" %(url))
        #os.system("tar -xvf CMPL.tar")

def getResultsForFilenamePart(results, fileNamePart):
    regex = re.compile(fileNamePart)
    files = results['filename'].tolist()
    match = [string for string in files if re.match(regex, string)]
    i=files.index(match[0])   
    return results[i]

def getImageURLs(casda_tap, sb):
    images = casda_tap.launch_job_async("SELECT * FROM ivoa.obscore WHERE obs_id='ASKAP-%s'" %(sb))
    r_images = images.get_results()

    results = []
    results.append(getResultsForFilenamePart(r_images, 'selavy-image.i.EMU_*.*.cont.taylor.0.restored.conv.components.xml'))
    results.append(getResultsForFilenamePart(r_images, 'selavy-image.i.EMU_*.*.cont.taylor.0.restored.conv.polarisation.xml'))
    results.append(getResultsForFilenamePart(r_images, 'image.restored.i.*.contcube.conv.fits'))
    results.append(getResultsForFilenamePart(r_images, 'image.restored.q.*.contcube.conv.fits'))
    results.append(getResultsForFilenamePart(r_images, 'image.restored.u.*.contcube.conv.fits'))
    results.append(getResultsForFilenamePart(r_images, 'image.restored.v.*.contcube.conv.fits'))
    results.append(getResultsForFilenamePart(r_images, 'image.i.*.cont.taylor.0.restored.conv.fits'))
    results.append(getResultsForFilenamePart(r_images, 'image.i.*.cont.taylor.1.restored.conv.fits'))
    #results.append(getResultsForFilenamePart(r_images, 'residual.i.*.contcube.fits'))
    #results.append(getResultsForFilenamePart(r_images, 'residual.q.*.contcube.fits'))
    #results.append(getResultsForFilenamePart(r_images, 'residual.u.*.contcube.fits'))
    #results.append(getResultsForFilenamePart(r_images, 'residual.v.*.contcube.fits'))
    results.append(getResultsForFilenamePart(r_images, 'residual.i.*.cont.taylor.0.fits'))
    results.append(getResultsForFilenamePart(r_images, 'residual.i.*.cont.taylor.1.fits'))
    image_urls = casda.stage_data(results, verbose=True)
    return image_urls


def getEvalURLs(casda_tap, sb):
    evalfiles = casda_tap.launch_job_async("SELECT * FROM casda.observation_evaluation_file oef inner join casda.observation o on oef.observation_id = o.id where o.sbid = '%s'" %(sb))
    r_eval = evalfiles.get_results()
    evals = []
    evals.append(getResultsForFilenamePart(r_eval, 'BeamwiseSourcefinding'))
    evals.append(getResultsForFilenamePart(r_eval, 'calibration-metadata-processing'))
    tar_urls = casda.stage_data(evals, verbose=True)
    return tar_urls



if __name__ == '__main__':
    
    username="jennifer.west@dunlap.utoronto.ca"
    
    casda_tap = Tap("https://casda.csiro.au/casda_vo_tools/tap")
    passw = getpass.getpass(str("Enter password for user "+username+": "))
    casda = Casda(user=username, password=passw)
    sbids = ["60585"]

    for sb in sbids:
        os.system("mkdir "+sb)
        imageURLs = getImageURLs(casda_tap, sb)
        evalURLs = getEvalURLs(casda_tap, sb)
        print(f'Downloading image files for SBID = {sb}')
        download(imageURLs, sb)
        print(f'Downloading evaluationa files for SBID = {sb}')
        download(evalURLs, sb)

        

