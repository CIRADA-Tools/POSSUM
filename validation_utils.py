"""
Utility functions for the validation scripts

"""
import numpy as np
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy import units as u
from astropy.wcs import WCS
import astropy.wcs.utils as wcs
import scipy.stats as stats
import astropy.io.fits as pf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb

import os
import pandas as pd

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


def mindists(distances):
    #takes array of distances from beam centers to points and returns the minimum distance per point and the beam number (or array ID) of that minimum 

    mindists=np.min(distances,axis=0)
    beam_num=np.argmin(distances,axis=0)

    return mindists,beam_num
    
def mosdistance(mos_ra, mos_dec, ra, dec):
    mos_coord=SkyCoord(mos_ra*u.deg,mos_dec*u.deg,frame='fk5')
    source_coord=SkyCoord(ra*u.deg,dec*u.deg,frame='fk5')
    distances = mos_coord.separation(source_coord).value
    return distances

def getSourceVals(RA, DEC, w, cube, allfreq):
    c = SkyCoord(RA, DEC, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=np.rint(x)
    Y=np.rint(y)
    
    hdu=pf.open(cube, memmap=True, mode='denywrite')
    if Y < 0 or X < 0: #Y > hdu[0].data.shape[1] or X > hdu[0].data.shape[2] or 
        #print "Source not in image!"  
        col=[]
        return col, allfreq
    
    col=hdu[0].data[:,0,int(Y),int(X)]
    hdu.close()
    #col = cube[:,int(Y),int(X)]

    flag=np.zeros(col.size)
    flag[np.isfinite(col)]=1
    flag[np.abs(col)>10]=0
    flag[col==0]=0
    #pdb.set_trace()
    col=np.where(is_outlier(col),np.nan,col)
    col=np.where(np.isfinite(col),col,np.nan)
    col=np.where(col==0,np.nan,col)
    col=np.where(np.abs(col)>1E5,np.nan,col)
    #col=col[flag==1]
    #allfreq=allfreq[flag==1]
    
    return col, allfreq

def getSourceStats(datacube, ra, dec, wcs, distances, beam_dist, beam_num, allfreq):

    Vals=[]
    Freq=[]
    median=[]
    sixteenth=[]
    eightyforth=[]
    stdevs=[]
    skews=[]
    kurtosi=[]
    dist=[]
    bdist=[]
    bnum=[]
    ra_0=[]
    dec_0=[]

    numSources=len(ra)
    
    #pdb.set_trace()
    for source in range(0, numSources):
        vals, sfreq=getSourceVals(ra[source], dec[source], wcs, datacube, allfreq)

        if len(vals) > 0: 
            Freq.append(sfreq)
            Vals.append(vals)
            median.append(np.nanmedian(vals))
            sixteenth.append(np.nanpercentile(vals,16))
            eightyforth.append(np.nanpercentile(vals,84))
            stdevs.append(np.nanstd(vals[vals<1]))
            dist.append(distances[source])
            bdist.append(beam_dist[source])
            bnum.append(beam_num[source])
            ra_0.append(ra[source])
            dec_0.append(dec[source])
            skews.append(stats.skew(vals))
            kurtosi.append(stats.kurtosis(vals))
    
    return Vals, Freq, np.array(median), np.array(sixteenth), np.array(eightyforth), np.array(stdevs), np.array(skews), np.array(kurtosi), np.array(dist), np.array(bdist), np.array(bnum), np.array(ra_0), np.array(dec_0)

def makeSourcePlot(coli, colq, colu, colv, t0, t0_val, t1_val, ifreq, qfreq, ufreq, vfreq, ra_n, dec_n, w, path, sb, name):
    in_pi=path+"/polIm/image.restored.pi.SB"+str(sb)+"."+str(name)+".fits"
    pi=pf.getdata(in_pi)

    logcoli=np.log(coli)
    logfreq=np.log(ifreq)
    idx = np.isfinite(logfreq) & np.isfinite(logcoli)
    z = np.polyfit(logfreq[idx], logcoli[idx], 1)
    p = np.poly1d(z)
    xp = np.linspace(logfreq.min(), logfreq.max(), logfreq.shape[0])
    label="T0="+str(t0_val)+"\nT1/T0="+str(t1_val)+"\nalpha(fit)="+'%.2f' %z[0]
    
    matplotlib.rc('xtick', labelsize=6) 
    matplotlib.rc('ytick', labelsize=6) 
    fig, ax = plt.subplots(figsize=(8,8))
    
    #fig.suptitle("("+'%.4f' %ra_n + '%.4f' %dec_n+")", fontsize=8)
    fig.suptitle(name, fontsize=8)
    
    plt.subplot(331)
    plt.loglog(ifreq, coli, '.', np.exp(xp), np.exp(p(xp)), '-')
    plt.title("Stokes I",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    
    plt.text(0.12, 1.0, label, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=6)

    plt.subplot(332)
    plt.title("Stokes V",fontsize=7)
    plt.xlabel("Frequency (Hz)",fontsize=6)
    plt.ylabel("Flux (Jy/beam)",fontsize=6)
    plt.semilogx(vfreq, colv, '.')
    
    #plt.subplot(333)
    # this is an inset axes over the main axes
    c = SkyCoord(ra_n, dec_n, frame='fk5',unit='deg')
    x, y = wcs.skycoord_to_pixel(c,w)
    X=int(np.rint(x))
    Y=int(np.rint(y))
    plt.axes([0.75, 0.81, .1, .1])
    plt.imshow(t0[0,Y-10:Y+10,X-10:X+10], origin='lower')
    plt.axis('off')
    plt.title("Stokes I",fontsize=6)
    plt.axes([0.75, .68, .1, .1])
    plt.imshow(pi, origin='lower')
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

def makeHistogram(vals, stokes, title, min_dist, max_dist, path): 
    vals=vals[np.isfinite(vals)]
    stdev=np.std(vals)
    histstats=[np.average(vals), np.median(vals), stats.skew(vals), stats.kurtosis(vals), stdev]
    minx=0
    maxx=3*stdev
    if stokes=="I":
        minx=-0.1
    if stokes=="V":
        minx=-stdev
        maxx=stdev    
    else:
        minx=-3*stdev
    fig, ax = plt.subplots()
    #pdb.set_trace()
    plt.hist(vals, bins="auto", range=[minx, maxx])
    plt.yscale('log')
    plt.title("Stokes "+stokes+": "+title)
    label="Average = "+str(histstats[0])+"\n"+"Median = "+str(histstats[1])+"\n"+"Skew = "+str(histstats[2])+"\n"+"Kurtosis = "+str(histstats[3])+"\n"+"St. dev = "+str(histstats[4])
    plt.text(0.65, 0.85, label, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, fontsize=8)
    plt.xlabel("Stokes "+stokes+" flux (Jy)")
    fig.savefig(path+"/plots/"+stokes+"_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png",bbox_inches='tight')
    #plt.show()
    plt.clf()
    plt.cla()
    plt.close(fig)
    return histstats

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
    piav=np.zeros((x,y))

    dataq=hduq[0].data[:,0,Y-r:Y+r,X-r:X+r]
    datau=hduu[0].data[:,0,Y-r:Y+r,X-r:X+r]
    #dataq = np.ma.array(dataQ, mask=np.isnan(dataQ))
    #datau = np.ma.array(dataU, mask=np.isnan(dataU))
    qsq=np.square(dataq)
    usq=np.square(datau)
    pi = np.sqrt(qsq+usq)
    piav[:,:]=np.nanmedian(pi, axis=0)

    hduq.close()
    hduu.close()   

    pf.writeto(path+"/polIm/image.restored.pi.SB"+str(sb)+"."+str(name)+".fits", piav , head, overwrite=True)
    pf.writeto(path+"/polIm/image.restored.q.SB"+str(sb)+".contcube."+str(name)+".fits", dataq , head, overwrite=True)
    pf.writeto(path+"/polIm/image.restored.u.SB"+str(sb)+".contcube."+str(name)+".fits", datau , head, overwrite=True)

def makeHTML(rootpath, sb, base, fluxmin, fluxmax, min_dist, max_dist):


    #rootpath=sys.argv[1]
    #sb=sys.argv[2] # 7216
    #base=sys.argv[3] #"SN1006
    path=rootpath+"selavy-cont-image.i."+base+".SB"+str(sb)+".cont.taylor.0.restored/"

    #fluxmin=np.float(sys.argv[4]) #min flux limit in mJy
    #fluxmax=np.float(sys.argv[5]) #max flux limit in mJy

    #distance limits
    #min_dist=np.float(sys.argv[6])
    #max_dist=np.float(sys.argv[7])


    fluxminJy=fluxmin/1000.
    fluxmaxJy=fluxmax/1000.

    #component catalogue
    selavycomp=path+"selavy-image.i."+base+".SB"+str(sb)+".cont.taylor.0.restored.components.txt"
    #RM synthesis products
    selavypol=path+"selavy-image.i."+base+".SB"+str(sb)+".cont.taylor.0.restored.polarisation.txt"

    pd.set_option('display.max_colwidth', -1)
    #read selavy input
    headerrows=2

    df_pol = pd.read_fwf(selavypol, skiprows=0,  header=[0,1], na_values=['-nan'])
    df_pol.columns = df_pol.columns.map('_'.join)
    df_pol.rename(index=str, columns={'component_id_Unnamed: 1_level_1':'component_id', 'component_name_Unnamed: 2_level_1': 'component_name', 'ra_hms_cont_Unnamed: 3_level_1':'ra_hms_cont', 'dec_dms_cont_Unnamed: 4_level_1':'dec_dms_cont', 'pol_frac_Unnamed: 36_level_1':'pol_frac', 'pol_frac_err_Unnamed: 37_level_1':'pol_frac_err'},inplace=True)

    df_pol["source"] = "<div class=\"hover_img\"><a href=\'plots/sources/" + df_pol["component_name"].astype(str) + ".png\'>" +df_pol["component_name"].astype(str) + "<span><img src=\"plots/sources/" + df_pol["component_name"].astype(str) + ".png\"> /></span></a></div>"
    df_pol.sort_values(by=["flux_I_median_[mJy/beam]"])
    cut_ra = df_pol.loc[(df_pol["flux_I_median_[mJy/beam]"] > fluxmin) & (df_pol["flux_I_median_[mJy/beam]"] < fluxmax)]["ra_deg_cont_[deg]"].values
    cut_dec = df_pol.loc[(df_pol["flux_I_median_[mJy/beam]"] > fluxmin) & (df_pol["flux_I_median_[mJy/beam]"] < fluxmax)]["dec_deg_cont_[deg]"].values
    numsources = len(cut_ra)

    #df.loc[df["component_name"]=='J150638-392026'][["component_name","flux_I_median","pol_peak"]]
    #df[["component_name","flux_I_median","pol_peak"]]

    list(df_pol)

    df2=df_pol[["source", "ra_deg_cont_[deg]", "dec_deg_cont_[deg]", "flux_I_median_[mJy/beam]","rms_I_[mJy/beam]", "pol_peak_[mJy/beam]",  "pol_peak_err_[mJy/beam]", "pol_frac", "fd_peak_fit_[rad/m2]", "fd_peak_fit_err_[rad/m2]", "pol_ang_ref_[deg]", "pol_ang_ref_err_[deg]", "pol_ang_zero_[deg]", "pol_ang_zero_err_[deg]",  "lambda_ref_sq_[m2]", "rmsf_fwhm_[rad/m2]", "flux_Q_median_[mJy/beam]", "rms_Q_[mJy/beam]", "flux_U_median_[mJy/beam]", "rms_U_[mJy/beam]", "flux_V_median_[mJy/beam]", "rms_V_[mJy/beam]"]]
    df3=df2.loc[(df2["flux_I_median_[mJy/beam]"] > fluxmin) & (df2["flux_I_median_[mJy/beam]"] < fluxmax)]
    #df3=df3.sort_values(by=["pol_peak"], ascending=False)
    df3=df3.sort_values(by=["flux_I_median_[mJy/beam]"], ascending=False)
    df3[["source","flux_I_median_[mJy/beam]","pol_peak_[mJy/beam]"]]

    html="pol_data_SB"+sb+".html"
    #os.system("rm "+html)
    title="Polarization properties for SB"+str(sb)
    f=open(html, 'a')
    f.write("<!doctype html><title>"+title+"</title>\n")

    f.write("<link rel=stylesheet type=text/css href='style.css'>\n")
    f.write("<div class=page>")
    f.write("<h3>Summary plots</h3>")
    f.write("<a href=\"plots/I_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\"><img src=\"plots/I_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\" height=300px></a>\n")
    f.write("<a href=\"plots/V_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\"><img src=\"plots/V_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\" height=300px></a>\n")
    f.write("<a href=\"plots/Q_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\"><img src=\"plots/Q_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\" height=300px></a>\n")
    f.write("<a href=\"plots/U_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\"><img src=\"plots/U_sources_from_"+str(min_dist)+"_to_"+str(max_dist)+".png\" height=300px></a>\n")
    f.write("<a href=\"plots/stokesV_vs_dist_from_mosaic_centre.png\"><img src=\"plots/stokesV_vs_dist_from_mosaic_centre.png\" height=300px></a>\n")
    f.write("<h3>Source properties (sources brighter than "+str(fluxminJy)+" Jy)</h3>")
    f.write("</div>")
    f.close()

    df3.to_html(open(html, 'a'), escape=False)