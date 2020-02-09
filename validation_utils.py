"""
Utility functions for the validation scripts

"""
import numpy as np
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy import units as u
from astropy.wcs import WCS
import astropy.wcs.utils as wcs
import scipy.stats as stats
from scipy import *
import astropy.io.fits as pf
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pdb
import re
from datetime import datetime
import time
import os
import pandas as pd

#from scount1 import *


plt.ioff()

def is_outlier(points, thresh=3):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh

def read_centers(filename):
    #assumes format of ID (x, y) hh:mm:ss.sss,+/-dd:mm:ss.ss
    data=np.genfromtxt(filename,skip_header=0,delimiter=';',dtype=str)
    centers=[]
    for i in range(data.shape[0]):
        aa=data[i].split('  ')
        if len(aa)==3:
            bb=aa[2].split(',')
        if len(aa)==4:
            bb=aa[3].split(',')
        #print i,len(bb),bb
        centers.append(bb[0]+' '+bb[1])
    return centers

def read_centers_new(filename):
    #assumes format of ID (x, y) hh:mm:ss.sss,+/-dd:mm:ss.ss
    data=np.genfromtxt(filename,skip_header=0,delimiter=';',dtype=str)
    centers=[]
    for i in range(data.shape[0]):
        aa=data[i].split('  ')
        if len(aa)==5:
            bb=aa[4].split(',')
        if len(aa)==6:
            bb=aa[5].split(',')
        #print i,len(bb),bb
        centers.append(bb[0]+' '+bb[1])
    return centers


def hms2skycoord(centers):
    #takes string in format "hh:mm:ss.sss +/-dd:mm:ss.ss" returns skycoordinate array
    coords=SkyCoord(centers,frame='fk5',unit=(u.hourangle,u.deg))

    return coords

def getpoints(ra=[],dec=[],filename=[],delimiter=','):
    #either reads in point positions from filename or takes ra, dec (in degrees) array and converts to skycoord array. filename should hae format ra (in degrees) , dec (in degrees), if not comma separated then set delimiter
    if len(ra)==0:
        ra,dec=np.genfromtxt(filename,delimiter=delimiter)
    points=SkyCoord(ra*u.deg,dec*u.deg,frame='fk5')
    
    return points

def dists(coords,points):
    #computes distance from coords to points in degrees
    distances=np.zeros((coords.size,points.size))
    for i in range(coords.size):
        distances[i,:]=coords[i].separation(points).value
    
    return distances
    
def angles(coords,points):
    #computes distance from coords to points in degrees
    posangle=np.zeros((coords.size,points.size))
    for i in range(coords.size):
        posangle[i,:]=coords[i].position_angle(points).degree
    
    return posangle

def mindists_and_angles(distances,posangles):
    #takes array of distances from beam centers to points and returns the minimum distance per point and the beam number (or array ID) of that minimum 

    mindists=np.min(distances,axis=0)
    pa=posangles[np.where(distances==mindists)]
    beam_num=np.argmin(distances,axis=0)

    return mindists,pa,beam_num
    
def getBeamDistances(footfile, ra, dec):
    beamCentres=read_centers(footfile)
    beamCoords=hms2skycoord(beamCentres)
    points1=getpoints(ra, dec)
    distances=dists(beamCoords,points1)
    posangles=angles(beamCoords,points1)
    beam_dist,pa,beam_num=mindists_and_angles(distances,posangles)
    return beam_dist,pa,beam_num


def mindists(distances):
    #takes array of distances from beam centers to points and returns the minimum distance per point and the beam number (or array ID) of that minimum 

    mindists=np.min(distances,axis=0)
    beam_num=np.argmin(distances,axis=0)

    return mindists,beam_num
    
def getMosaicDistances(mos_ra, mos_dec, ra, dec):
    #centre of the mosaic
    points=getpoints(ra,dec)
    mos_coord=SkyCoord(mos_ra*u.deg,mos_dec*u.deg,frame='fk5')
    source_coord=SkyCoord(ra*u.deg,dec*u.deg,frame='fk5')
    posangle=mos_coord.position_angle(points).degree
    distances = mos_coord.separation(source_coord).value
    return distances, posangle


def getSourceVals(RA, DEC, w, hdu, allfreq):
    c = SkyCoord(RA, DEC, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=np.rint(x)
    Y=np.rint(y)
    
    #hdu=pf.open(cube, memmap=True, mode='denywrite')
    if Y < 0 or X < 0: #Y > hdu[0].data.shape[1] or X > hdu[0].data.shape[2] or 
        #print "Source not in image!"  
        col=[]
        return col, allfreq
    
    col=hdu[0].data[:,0,int(Y),int(X)]
    #hdu.close()
    #col = cube[:,int(Y),int(X)]

    flag=np.zeros(col.size)
    flag[np.isfinite(col)]=1
    flag[np.abs(col)>10]=0
    flag[col==0]=0
    #pdb.set_trace()
    col=np.where(is_outlier(col),np.nan,col)
    col=np.where(np.isfinite(col),col,np.nan)
    col=np.where(col==0,np.nan,col)
    col=np.where(np.abs(col)>1E15,np.nan,col)
    #col=col[flag==1]
    #allfreq=allfreq[flag==1]
    
    return col, allfreq
    
def getSourceCutout(RA, DEC, w, hdu, allfreq, imsize):
    c = SkyCoord(RA, DEC, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=int(np.rint(x))
    Y=int(np.rint(y))
    
    #hdu=pf.open(cube, memmap=True, mode='denywrite')
    if Y < 0 or X < 0: #Y > hdu[0].data.shape[1] or X > hdu[0].data.shape[2] or 
        #print "Source not in image!"  
        col=[]
        return col, allfreq
    
    col=hdu[0].data[:,0,int(Y),int(X)]
    
    y = imsize
    x = imsize
    r = int(np.rint(imsize/2))
    r2= int(np.rint(r/2))

    data=hdu[0].data[:,0,Y-r:Y+r,X-r:X+r]
    innerpix=hdu[0].data[:,0,Y-r2:Y+r2,X-r2:X+r2]
    edgepix=data - np.pad(innerpix, ((0,0), (r-r2,r-r2), (r-r2,r-r2) ), mode='constant') 

    
    #hdu.close()
    #col = cube[:,int(Y),int(X)]

    flag=np.zeros(col.size)
    flag[np.isfinite(col)]=1
    flag[np.abs(col)>10]=0
    flag[col==0]=0
    #pdb.set_trace()
    col=np.where(is_outlier(col),np.nan,col)
    col=np.where(np.isfinite(col),col,np.nan)
    col=np.where(col==0,np.nan,col)
    col=np.where(np.abs(col)>1E15,np.nan,col)
    #col=col[flag==1]
    #allfreq=allfreq[flag==1]
    
    return col, data, edgepix, allfreq

def getSourceStats(datacube, ra, dec, wcs, allfreq, imsize):

    numSources=len(ra)

    Vals=[None] * numSources
    DataCutouts=[None] * numSources
    EdgePix=[None] * numSources
    Freq=[None] * numSources
    median=[None] * numSources
    sixteenth=[None] * numSources
    eightyforth=[None] * numSources
    stdevs=[None] * numSources
    skews=[None] * numSources
    kurtosi=[None] * numSources
    dist=[None] * numSources
    bdist=[None] * numSources
    bnum=[None] * numSources
    ra_0=[None] * numSources
    dec_0=[None] * numSources


    hdu=pf.open(datacube, memmap=True, mode='denywrite')
    #pdb.set_trace()
    for source in range(0, numSources):
        vals, data, edgepix, sfreq=getSourceCutout(ra[source], dec[source], wcs, hdu, allfreq, imsize)
        uncertainty = np.median(edgepix,axis=(1, 2))
        if len(vals) > 0: 
            
            Freq[source]=sfreq
            Vals[source]=vals
            DataCutouts[source]=data
            EdgePix[source]=uncertainty
            median[source]=np.nanmedian(vals)
            sixteenth[source]=np.nanpercentile(vals,16)
            eightyforth[source]=np.nanpercentile(vals,84)
            stdevs[source]=np.nanstd(vals[vals<1])
            ra_0[source]=ra[source]
            dec_0[source]=dec[source]
            skews[source]=stats.skew(vals)
            kurtosi[source]=stats.kurtosis(vals)

            
    hdu.close()
    
    #sourcestats=[Vals, DataCutouts, EdgePix, Freq, np.array(median), np.array(sixteenth), np.array(eightyforth), np.array(stdevs), np.array(skews), np.array(kurtosi), np.array(dist), np.array(bdist), np.array(bnum), np.array(ra_0), np.array(dec_0)]
    sourcestats={'Vals':         Vals, 
            'Data':  DataCutouts, 
            'EdgePix':      EdgePix, 
            'Freq':         Freq, 
            'Medians':      np.array(median), 
            'Sixteenth':    np.array(sixteenth), 
            'EightyForth':  np.array(eightyforth), 
            'StDev':        np.array(stdevs), 
            'Skew':         np.array(skews), 
            'Kurtosis':     np.array(kurtosi), 
            'RA':           np.array(ra_0), 
            'Dec':          np.array(dec_0)}

    return sourcestats


def makePIImage(path, sb, in_q, in_u, ra_n, dec_n, name, imsize):

    hduq=pf.open(in_q, memmap=True, mode='denywrite')
    hduu=pf.open(in_u, memmap=True, mode='denywrite')

    # this is an inset axes over the main axes
    w = WCS(in_q)
    c = SkyCoord(ra_n, dec_n, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=int(np.rint(x))
    Y=int(np.rint(y))
    head=pf.getheader(in_q)
    y = imsize
    x = imsize
    r = int(np.rint(imsize/2))

    dataq=hduq[0].data[:,0,Y-r:Y+r,X-r:X+r]
    datau=hduu[0].data[:,0,Y-r:Y+r,X-r:X+r]
    
    #dataq = np.ma.array(dataQ, mask=np.isnan(dataQ))
    #datau = np.ma.array(dataU, mask=np.isnan(dataU))
    qsq=np.square(dataq)
    usq=np.square(datau)
    pi = np.sqrt(qsq+usq)
    piav=np.zeros((x,y))
    piav[:,:]=np.nanmedian(pi, axis=0)

    hduq.close()
    hduu.close()  
    head["CRVAL1"]=ra_n
    head["CRVAL2"]=dec_n
    head["CRPIX1"]=imsize/2
    head["CRPIX2"]=imsize/2 

    pf.writeto(path+"/polIm/image.restored.pi.SB"+str(sb)+"."+str(name)+".fits", piav , head, overwrite=True)
    pf.writeto(path+"/polIm/image.restored.q.SB"+str(sb)+".contcube."+str(name)+".fits", dataq , head,  overwrite=True)
    pf.writeto(path+"/polIm/image.restored.u.SB"+str(sb)+".contcube."+str(name)+".fits", datau , head, overwrite=True)

def makePIImageFromData(path, sb, dataq, datau, ra_n, dec_n, type, name, head, imsize):

    qsq=np.square(dataq)
    usq=np.square(datau)
    pi = np.sqrt(qsq+usq)
    piav=np.zeros((imsize,imsize))
    piav[:,:]=np.nanmedian(pi, axis=0) 
    #pdb.set_trace()
    head["CRVAL1"]=ra_n
    head["CRVAL2"]=dec_n
    head["CRPIX1"]=imsize/2
    head["CRPIX2"]=imsize/2
    pf.writeto(path+"/polIm/"+type+".pi.SB"+str(sb)+"."+str(name)+".fits", piav , head, overwrite=True)
    pf.writeto(path+"/polIm/"+type+".q.SB"+str(sb)+".contcube."+str(name)+".fits", dataq , head, overwrite=True)
    pf.writeto(path+"/polIm/"+type+".u.SB"+str(sb)+".contcube."+str(name)+".fits", datau , head, overwrite=True)

def makeIV_Cutouts(path, sb, datai, datav, ra_n, dec_n, type, name, head, imsize):

    head["CRVAL1"]=ra_n
    head["CRVAL2"]=dec_n
    head["CRPIX1"]=imsize/2
    head["CRPIX2"]=imsize/2
    pf.writeto(path+"/polIm/"+type+".i.SB"+str(sb)+"."+str(name)+".fits", datai , head, overwrite=True)
    pf.writeto(path+"/polIm/"+type+".v.SB"+str(sb)+".contcube."+str(name)+".fits", datav , head, overwrite=True)


def makeSourcePlot(coli, colq, colu, colv, t0, t0_val, t1_val, ifreq, qfreq, ufreq, vfreq, ra_n, dec_n, w, path, sb, name):
    #pdb.set_trace()
    in_pi=path+"/polIm/image.restored.pi.SB"+str(sb)+"."+str(name)+".fits"
    pi=pf.getdata(in_pi)

    logcoli=np.log(coli)
    logfreq=np.log(ifreq)
    idx = np.isfinite(logfreq) & np.isfinite(logcoli)
    fig, ax = plt.subplots(figsize=(9,9))

    if(len(logfreq[idx]) != 0):
        z = np.polyfit(logfreq[idx], logcoli[idx], 1)
        p = np.poly1d(z)
        xp = np.linspace(logfreq.min(), logfreq.max(), logfreq.shape[0])
        label="T0="+str(t0_val)+"\nT1/T0="+str(t1_val)+"\nalpha(fit)="+'%.2f' %z[0]
    
        matplotlib.rc('xtick', labelsize=6)
        matplotlib.rc('ytick', labelsize=6) 
    
        #fig.suptitle("("+'%.4f' %ra_n + '%.4f' %dec_n+")", fontsize=8)
        fig.suptitle(name, fontsize=8)
    
        plt.subplot(331)        
        plt.loglog(ifreq, coli, '.', np.exp(xp), np.exp(p(xp)), '-')
        plt.title("Stokes I",fontsize=7)
        plt.xlabel("Frequency (Hz)",fontsize=6)
        plt.ylabel("Flux (Jy/beam)",fontsize=6)
        #plt.xticks(np.arange(np.min(ifreq), np.max(ifreq), (np.max(ifreq)-np.min(ifreq))/4 ))
        plt.text(0.12, 1.0, label, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=6)
        

    ax2=plt.subplot(332)
    plt.title("Stokes V",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    #plt.xticks(np.arange(np.min(ifreq), np.max(ifreq), (np.max(ifreq)-np.min(ifreq))/4 ))
    #ax2.set_xticks(ax.get_xticks()[::2])
    plt.semilogx(vfreq, colv, '.')
    
    #plt.scatter(vfreq, colv, marker='.')
    
    #plt.subplot(333)
    # this is an inset axes over the main axes
    c = SkyCoord(ra_n, dec_n, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=int(np.rint(x))
    Y=int(np.rint(y))
    plt.axes([0.75, 0.81, .1, .1])    
    plt.imshow(t0[0,Y-10:Y+10,X-10:X+10], origin='lower',zorder=1)  
    plt.plot(10.,10.,'+',zorder=2,color="red")
    plt.axis('off')
    plt.title("Stokes I",fontsize=6)
    plt.axes([0.75, .68, .1, .1])    
    plt.imshow(pi, origin='lower',zorder=1)
    plt.plot(10.,10.,'+',zorder=2,color="red")
    plt.axis('off')
    plt.title("Polarized Intensity",fontsize=6)
    #plt.xticks([])
    #plt.yticks([])


    plt.subplot(334)
    plt.title("Stokes Q",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(qfreq, colq, '.')

    plt.subplot(335)
    plt.title("Stokes U",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(ufreq, colu, '.')
    #pdb.set_trace()    
    plt.subplot(336)
    plt.title("Stokes Q vs U",fontsize=7)
    plt.xlabel("Flux (Jy/beam)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    if (len(qfreq)==len(ufreq)):
        plt.scatter(colq, colu, c=ufreq, cmap=plt.cm.rainbow, s=1)
        freq=qfreq
    if (len(qfreq)>len(ufreq)):
        diff=np.setdiff1d(qfreq,ufreq,assume_unique=True)
        index=[]
        for i in range(0, len(diff)): 
            index.append(np.where(qfreq==diff[i])[0][0])
            colq=np.delete( colq, np.array(index))
        
        #colq=colq[colq!=colq[np.where(qfreq==diff)]]
        freq=ufreq
        plt.scatter(colq, colu, c=ufreq, cmap=plt.cm.rainbow, s=1)
    if (len(qfreq)<len(ufreq)):
        diff=np.setdiff1d(ufreq,qfreq,assume_unique=True)
        index=[]
        for i in range(0, len(diff)): 
            index.append(np.where(ufreq==diff[i])[0][0])
            colu=np.delete( colu, np.array(index))

        #colu=colu[colu!=colu[np.where(ufreq==diff)]]
        plt.scatter(colq, colu, c=qfreq, cmap=plt.cm.rainbow, s=1)
        freq=qfreq
        
        

    plt.subplot(337)
    plt.title("Polarized intensity",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(freq, np.sqrt(colq*colq + colu*colu), '.')
    
    plt.subplot(338)
    plt.title("Polarized fraction",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("PI/Stokes I",fontsize=6)
    plt.semilogx(freq, np.sqrt(colq*colq + colu*colu)/coli, '.')

    plt.subplot(339)
    plt.title("Leakage" ,fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Stokes V/Stokes I",fontsize=6)
   
    if (len(vfreq)==len(ifreq)):
        freq=vfreq
        plt.semilogx(freq, colv/coli, '.')
    if (len(vfreq)>len(ifreq)):
        diff=np.setdiff1d(vfreq,ifreq,assume_unique=True)
        index=[]
        for i in range(0, len(diff)): 
            index.append(np.where(vfreq==diff[i])[0][0])
            colv=np.delete( colv, np.array(index))
        #colv=colv[colv!=colv[np.where(vfreq==diff)]]
        freq=ifreq
        plt.semilogx(freq, colv/coli, '.')
    if (len(vfreq)<len(ifreq)):
        diff=np.setdiff1d(ifreq,vfreq,assume_unique=True)
        index=[]
        for i in range(0, len(diff)): 
            index.append(np.where(ifreq==diff[i])[0][0])
            coli=np.delete( coli, np.array(index))

        freq=vfreq
        plt.semilogx(freq, colv/coli, '.')
    

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    #plt.show()

    #fig.savefig(path+"plots/sources/source_"+str(ra_n)+"_"+str(dec_n)+".png",bbox_inches='tight')
    fig.savefig(path+"plots/sources/"+name+".png",bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close(fig)

def makeSourcePlotForResiduals(coli, colq, colu, colv, erri, errq, erru, errv, t0, ifreq, qfreq, ufreq, vfreq, ra_n, dec_n, w, path, sb, type, name):
    #pdb.set_trace()
    in_pi=path+"/polIm/"+type+".pi.SB"+str(sb)+"."+str(name)+".fits"

    pi=pf.getdata(in_pi)

    logcoli=np.log(coli)
    logfreq=np.log(ifreq)
    idx = np.isfinite(logfreq) & np.isfinite(logcoli)
    fig, ax = plt.subplots(figsize=(8,8))

    if(len(logfreq[idx]) != 0):
        z = np.polyfit(logfreq[idx], logcoli[idx], 1)
        p = np.poly1d(z)
        xp = np.linspace(logfreq.min(), logfreq.max(), logfreq.shape[0])
        label="alpha(fit)="+'%.2f' %z[0]
    
        matplotlib.rc('xtick', labelsize=6) 
        matplotlib.rc('ytick', labelsize=6) 
    
        #fig.suptitle("("+'%.4f' %ra_n + '%.4f' %dec_n+")", fontsize=8)
        fig.suptitle(name, fontsize=8)
    
        plt.subplot(331)
        plt.loglog(ifreq, coli, '.', np.exp(xp), np.exp(p(xp)), '-')
        plt.title("Stokes I - on source",fontsize=7)
        plt.xlabel("Frequency (Hz)",fontsize=6)
        plt.ylabel("Flux (Jy/beam)",fontsize=6)
    
        plt.text(0.12, 1.0, label, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=6)

    plt.subplot(332)
    plt.title("Stokes I - off source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(ifreq, erri, '.')
    
    #plt.subplot(333)
    # this is an inset axes over the main axes
    c = SkyCoord(ra_n, dec_n, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=int(np.rint(x))
    Y=int(np.rint(y))
    plt.axes([0.75, 0.81, .1, .1])    
    plt.imshow(t0[0,Y-10:Y+10,X-10:X+10], origin='lower',zorder=1)  
    plt.plot(10.,10.,'+',zorder=2,color="red")
    plt.axis('off')
    plt.title("Stokes I",fontsize=6)
    plt.axes([0.75, .68, .1, .1])    
    plt.imshow(pi, origin='lower',zorder=1)
    plt.plot(10.,10.,'+',zorder=2,color="red")
    plt.axis('off')
    plt.title("Polarized Intensity",fontsize=6)
    #plt.xticks([])
    #plt.yticks([])


    plt.subplot(334)
    plt.title("Stokes Q - on source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(qfreq, colq, '.')

    plt.subplot(335)
    plt.title("Stokes U - on source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(ufreq, colu, '.')

    plt.subplot(336)
    plt.title("Stokes V - on source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(vfreq, colv, '.')

    plt.subplot(337)
    plt.title("Stokes Q - off source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(qfreq, errq,'.')
    
    plt.subplot(338)
    plt.title("Stokes U - off source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(ufreq, erru,'.')

    plt.subplot(339)
    plt.title("Stokes V - off source",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(vfreq, errv,'.')
   
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    #plt.show()

    #fig.savefig(path+"plots/sources/source_"+str(ra_n)+"_"+str(dec_n)+".png",bbox_inches='tight')
    fig.savefig(path+"plots/sources/"+type+"."+name+".png",bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close(fig)


def makeHistogram(vals, stokes, title, plotname):
    vals=vals[np.isfinite(vals)]
    stdev=np.nanstd(vals)
    histstats=[np.nanmean(vals), np.nanmedian(vals), stats.skew(vals), stats.kurtosis(vals), stdev]

    minx=0
    
    if stokes=="Stokes I":
        minx=-0.01
        maxx=3*stdev
    elif stokes=="Stokes V":
        minx=-2*stdev
        maxx=2*stdev    
    elif stokes=="Leakage":
        minx=0.0
        maxx=4*stdev
    else:
        minx=-5*stdev
        maxx=5*stdev
    fig, ax = plt.subplots()

    plt.hist(vals, bins="auto", range=[minx, maxx])
    plt.yscale('log')
    plt.title(stokes+": "+title)
    label="Average = "+"{0:.2f}".format(histstats[0])+"\n"+"Median = "+"{0:.2f}".format(histstats[1])+"\n"+"Skew = "+"{0:.2f}".format(histstats[2])+"\n"+"Kurtosis = "+"{0:.2f}".format(histstats[3])+"\n"+"St. dev = "+"{0:.2f}".format(histstats[4])
    plt.text(0.65, 0.85, label, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.xlabel("Stokes "+stokes+" flux (Jy)")
    fig.savefig(plotname,bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close(fig)
    return histstats
    

def makeVvsDist(v_dists, v_sixteenth, v_eightyforth, v_medians, v_stdevs, xlabel, fluxmin, fluxmax, outfile):
    #Stokes V as a function of distance from the mosaic centre
    fig, ax = plt.subplots()
    plt.fill_between(v_dists[np.argsort(v_dists)], v_sixteenth[np.argsort(v_dists)], v_eightyforth[np.argsort(v_dists)], alpha=0.5)
    lm=plt.plot(v_dists[np.argsort(v_dists)],v_medians[np.argsort(v_dists)], color='blue')
    lsd=plt.plot(v_dists[np.argsort(v_dists)],v_stdevs[np.argsort(v_dists)], '-', color='orange')
    #plt.ylim(-0.005, 0.005)
    plt.ylabel("Stokes V")
    #plt.yscale('log')
    plt.xlabel(xlabel)
    plt.title("Sources from "+str(fluxmin)+" to "+str(fluxmax)+" Jy")
    blue_patch = mpatches.Patch(color='blue', label='Median')
    orange_patch = mpatches.Patch(color='orange', label='St Dev')
    plt.legend(handles=[blue_patch, orange_patch])
    #plt.figlegend( (lm, lsd),('median', 'stdev'),'upper right' )
    fig.savefig(outfile,bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close(fig)
    
def makeLeakagePlot(pa, beam_num, beam_dist, leakages, plotfilename):
    #Leakage as a function of position in the beam
    #fig, ax = plt.subplots()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    pa_rad=np.radians(pa)
    colors = beam_num/36.*np.pi*2.

    #ax.set_rticklabels([])
    c = ax.scatter(pa_rad,beam_dist,c=colors,s=leakages*500,alpha=0.5,cmap='hsv')
    ax.set_rticks([0.4, 0.6, 0.8])
    #c = ax.scatter(pa_rad,beam_dist,c="red",s=leakages*1000000,alpha=0.9)
    plt.title("Leakage")
    fig.savefig(plotfilename,bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close(fig)
    
def makeLogNLogP(df, nbins, area, fluxmin, fluxmax, plotfilename):
    intcounts,interr,binscx = justint(df["pol_peak"].values,nbins, area)#30,36)
    #print("don't hit continue! check plots!")
    fig, ax = plt.subplots()
    #plt.loglog(intcounts,binscx, '-')
    plt.errorbar(np.log(intcounts),np.log(binscx),yerr=interr)
    plt.ylabel("Log N")
    #plt.yscale('log')
    plt.xlabel("Log P")
    plt.title("Sources from "+str(fluxmin)+" to "+str(fluxmax)+" mJy")
    fig.savefig(plotfilename,bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close(fig)

#Adapted from code provided by Tessa Vernstrom    
def justint(vals,nbins,area):
    """ calulate integral source count over nbins logged spaced bins, assuming equal area for all fluxes"""
    minb=min(vals)*.99
    maxb=max(vals)*1.01
    maxbl=log10(maxb)
    minbl=log10(minb)

    bins=np.logspace(minbl,maxbl,nbins+1)
    width1=bins[1:]-bins[:-1]
    binscx=(bins[1:]+bins[:-1])/2.

    intcounts=np.zeros(nbins)
    interr=np.zeros(nbins)
    for i in range(nbins):
        intcounts[i]=np.argwhere(vals>=bins[i]).size
        interr[i]=sqrt(intcounts[i])
    intcounts=intcounts/area
    interr=interr/area
    
    return intcounts,interr,binscx


def makeScatter(xdata, ydata, dx, dy, xmin, xmax, ymin, ymax, xlabel, ylabel, title, fileout):

    fig, ax = plt.subplots()
    #plt.scatter(xdata, ydata, alpha=0.6, s=5)
    plt.errorbar(xdata, ydata, alpha=0.6, xerr=dx ,yerr=dy,fmt='+')
    plt.ylim(ymin, ymax)
    plt.xlim(xmin,xmax)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    fig.savefig(fileout,bbox_inches='tight')
    
def getSolarFlux(obsdate):
    link="ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt"
    mean=0
    std=0
    jdmean=0
    
    try:
        
        fluxtable=pd.read_fwf(link, header=[0], skiprows=[1], timeout=60.0)
        #time.sleep(10)
        #obsdate=20191016
        obs=fluxtable[fluxtable.fluxdate==obsdate[0]]
        solarflux=obs.fluxobsflux
        mean=solarflux.mean()
        std=solarflux.std()
        jdmean=obs.fluxjulian.mean()
    except:
        pass
        
    return mean, std, jdmean
    
def getObsVar(mslist):
    f=open(mslist, "r")
    lines=f.read().splitlines()
    line4=re.split('\s+', lines[4])
    line8=re.split('\s+', lines[8])
    line11=re.split('\s+', lines[11])
   
    ra=line8[6]
    dec=line8[7]

    freq0=line11[8]
    chanWid=line11[9]
    ctrFreq=line11[11]

    obsStart=line4[6]
    obsEnd=line4[8]

    date=re.split('/+', obsStart)[0]
    date_object = datetime.strptime(date, "%d-%b-%Y")
    obsdate=date_object.strftime("%Y%m%d")
    obsVars=[ra,dec,freq0,chanWid,ctrFreq,obsStart, obsEnd, int(obsdate)]
    return obsVars
    
def getOppermannAv(opDataFile, opErrFile, beamCoords):
    Op_data=hp.read_map(opDataFile)
    Op_data_err=hp.read_map(opErrFile)
    Op_fd=[]
    Op_fd_err=[]
    for i in range(beamCoords.size):
        coord_gal=beamCoords[i].transform_to('galactic')
        pix_no=hp.ang2pix(128, coord_gal.l.deg, coord_gal.b.deg, lonlat=True)
        Op_fd.append(Op_data[pix_no])
        Op_fd_err.append(Op_data_err[pix_no])
    return np.mean(np.array(Op_fd)), np.mean(np.array(Op_fd_err))
    
def getFrequencies(inFITS):
    head=pf.getheader(inFITS)
    #get frequencies
    if head['ctype3']=='FREQ': freqAx=3
    elif head['ctype4']=='FREQ': freqAx=4

    nFreq = head["NAXIS"+str(freqAx)]
    cdelt = head['cdelt'+str(freqAx)]
    crpix = head['crpix'+str(freqAx)]
    crval = head['crval'+str(freqAx)]
    freq = np.arange(nFreq, dtype = np.float32)
    for j in range(nFreq):
        freq[j] = (j + 1 - crpix) * cdelt + crval
    return freq

def makePlotsForSources(filtered, iStats, qStats, uStats, vStats, taylor0, taylor1, path, sb, head, imsize):

    #plot spectrum for every source
    
    t0 =pf.getdata(taylor0)[:,0,:,:]
    t1 = pf.getdata(taylor1)[:,0,:,:]

    
    w0 = WCS(taylor0)
    c0 = SkyCoord(filtered["ra_deg_cont"].values, filtered["dec_deg_cont"].values, frame='fk5',unit='deg')
    x0, y0 = wcs.skycoord_to_pixel(c0,w0)
    X0=np.rint(x0)
    Y0=np.rint(y0)

    t0_val=t0[:,np.int_(Y0),np.int_(X0)]
    t1_val=t1[:,np.int_(Y0),np.int_(X0)]/t0_val
    names=filtered['component_name'].values
    numSources=len(names)
  
    for source in range(0, numSources):
        print("Making plot for "+str(names[source])+" ("+str(source+1)+" of "+ str(numSources) +").")
        makePIImageFromData(path, sb, qStats['Data'][source], uStats['Data'][source], iStats['RA'][source], iStats['Dec'][source], "image.restored", names[source], head, imsize)
        makeIV_Cutouts(path, sb, iStats['Data'][source], vStats['Data'][source], iStats['RA'][source], iStats['Dec'][source], "image.restored", names[source], head, imsize)

        makeSourcePlot( iStats['Vals'][source], qStats['Vals'][source], uStats['Vals'][source], vStats['Vals'][source], t0, t0_val[0][source], t1_val[0][source],\
                       iStats['Freq'][source], qStats['Freq'][source], uStats['Freq'][source], vStats['Freq'][source], iStats['RA'][source], iStats['Dec'][source], w0, path, sb, names[source] )

def makePlotsForResiduals(filtered, iStats, qStats, uStats, vStats, taylor0, type, path, sb, head, imsize):
    #plot spectrum for every source
    t0 = pf.getdata(taylor0)[:,0,:,:]
    names=filtered['component_name'].values
    numSources=len(names)
    w0 = WCS(taylor0)
    for source in range(0, numSources):
        print("Making plot for "+type+" "+str(names[source])+" ("+str(source+1)+" of "+ str(numSources) +").")
        makePIImageFromData(path, sb, qStats['Data'][source], uStats['Data'][source], iStats['RA'][source], iStats['Dec'][source], type, names[source], head, imsize)
        makeIV_Cutouts(path, sb, iStats['Data'][source], vStats['Data'][source], iStats['RA'][source], iStats['Dec'][source], type, names[source], head, imsize)
        makeSourcePlotForResiduals(iStats['Vals'][source], qStats['Vals'][source], uStats['Vals'][source], vStats['Vals'][source], iStats['EdgePix'][source],\
                                   qStats['EdgePix'][source], uStats['EdgePix'][source], vStats['EdgePix'][source], t0, iStats['Freq'][source], qStats['Freq'][source],uStats['Freq'][source],\
                                   vStats['Freq'][source], iStats['RA'][source], iStats['Dec'][source], w0, path, sb, type, names[source])
