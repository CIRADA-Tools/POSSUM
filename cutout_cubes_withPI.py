"""
Makes a cutout of a set of image cubes. Also makes a PI image cube for the same region.

"""
import sys
import numpy as np
import astropy.io.fits as pf
from astropy.wcs import WCS
import astropy.wcs.utils as wcs
from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames


path=sys.argv[1]
sb=sys.argv[2] # 7216
base=sys.argv[3] #"SN1006
ra_n=sys.argv[4]
dec_n=sys.argv[5]
size=int(sys.argv[6])
in_i= path+"image.restored.i."+base+".SB"+str(sb)+".contcube.fits" #image.restored.i.SN1006.SB7216.contcube.fits"
in_q= path+"image.restored.q."+base+".SB"+str(sb)+".contcube.fits" #image.restored.q.SN1006.SB7216.contcube.fits"
in_u= path+"image.restored.u."+base+".SB"+str(sb)+".contcube.fits" #"image.restored.u.SN1006.SB7216.contcube.fits"

hdui=pf.open(in_i, memmap=True, mode='denywrite')
hduq=pf.open(in_q, memmap=True, mode='denywrite')
hduu=pf.open(in_u, memmap=True, mode='denywrite')
w = WCS(in_q)
# this is an inset axes over the main axes
c = SkyCoord(ra_n, dec_n, frame='fk5',unit='deg')
x, y = wcs.skycoord_to_pixel(c,w)
X=int(np.rint(x))
Y=int(np.rint(y))

#plt.imshow(t0[0,Y-10:Y+10,X-10:X+10], origin='lower')
#get header and data
head=pf.getheader(in_q)
y = size
x = size
piav=np.zeros((x,y))

datai=hdui[0].data[:,0,int(Y-(size/2)):int(Y+(size/2)),int(X-(size/2)):int(X+(size/2))]
dataq=hduq[0].data[:,0,int(Y-(size/2)):int(Y+(size/2)),int(X-(size/2)):int(X+(size/2))]
datau=hduu[0].data[:,0,int(Y-(size/2)):int(Y+(size/2)),int(X-(size/2)):int(X+(size/2))]
#dataq = np.ma.array(dataq, mask=np.isnan(dataq))
#datau = np.ma.array(datau, mask=np.isnan(datau))
qsq=np.square(dataq)
usq=np.square(datau)
pi = np.sqrt(qsq+usq)
piav[:,:]=np.nanmedian(pi, axis=0)

hduq.close()
hduu.close()   

head['CRVAL1'] = ra_n
head['CRPIX1'] = int(x/2)
head['CRVAL2'] = dec_n
head['CRPIX2'] = int(y/2)

pf.writeto(path+"image.restored.pi.SB"+str(sb)+"_"+ra_n+"."+dec_n+".fits", piav , head, overwrite=True)
pf.writeto(path+"image.restored.i.SB"+str(sb)+".contcube_"+ra_n+"."+dec_n+".fits", datai , head, overwrite=True)
pf.writeto(path+"image.restored.q.SB"+str(sb)+".contcube_"+ra_n+"."+dec_n+".fits", dataq , head, overwrite=True)
pf.writeto(path+"image.restored.u.SB"+str(sb)+".contcube_"+ra_n+"."+dec_n+".fits", datau , head, overwrite=True)