import os
import pandas as pd
import re
import numpy as np
import datetime
from astropy.wcs import WCS
import validation_utils as vu
import pickle
import pdb

#Define the names and paths of the files needed. Returns a dictionary of the filenames.
#this is currently using the ASKAPpipeline naming convention
def pipelineFiles(path, base, sb):

    #Define file names

    #restored image cubes
    in_i= "image.restored.i."+base+"SB"+str(sb)+".contcube.fits" #image.restored.i.SB4612.contcube.fits"
    in_q= "image.restored.q."+base+"SB"+str(sb)+".contcube.fits" #image.restored.q.SB4612.contcube.fits"
    in_u= "image.restored.u."+base+"SB"+str(sb)+".contcube.fits" #"image.restored.u.SB4612.contcube.fits"
    in_v= "image.restored.v."+base+"SB"+str(sb)+".contcube.fits" #"image.restored.v.SB4612.contcube.fits"

    #residual image cubes
    res_i= "residual.i."+base+"SB"+str(sb)+".contcube.fits" #image.restored.i.SB4612.contcube.fits"
    res_q= "residual.q."+base+"SB"+str(sb)+".contcube.fits" #image.restored.q.SB4612.contcube.fits"
    res_u= "residual.u."+base+"SB"+str(sb)+".contcube.fits" #"image.restored.u.SB4612.contcube.fits"
    res_v= "residual.v."+base+"SB"+str(sb)+".contcube.fits" #"image.restored.v.SB4612.contcube.fits"

    #weight cubes
    w_i= "weights.i."+base+"SB"+str(sb)+".contcube.fits" #image.restored.i.SB4612.contcube.fits"
    w_q= "weights.q."+base+"SB"+str(sb)+".contcube.fits" #image.restored.q.SB4612.contcube.fits"
    w_u= "weights.u."+base+"SB"+str(sb)+".contcube.fits" #"image.restored.u.SB4612.contcube.fits"
    w_v= "weights.v."+base+"SB"+str(sb)+".contcube.fits" #"image.restored.v.SB4612.contcube.fits"

    # taylor term mfs images
    taylor0=path+"image.i."+base+"SB"+str(sb)+".cont.taylor.0.restored.fits" 
    taylor1=path+"image.i."+base+"SB"+str(sb)+".cont.taylor.1.restored.fits" 
    restaylor0=path+"residual.i."+base+"SB"+str(sb)+".cont.taylor.0.fits" 
    restaylor1=path+"residual.i."+base+"SB"+str(sb)+".cont.taylor.1.fits" 
    wtaylor0=path+"weights.i."+base+"SB"+str(sb)+".cont.taylor.0.fits" 
    wtaylor1=path+"weights.i."+base+"SB"+str(sb)+".cont.taylor.1.fits" 


    #selavy output (Stokes I components and polarization)
    selavy=path+"selavy-cont-image.i."+base+"SB"+str(sb)+".cont.taylor.0.restored/selavy-image.i."+base+"SB"+str(sb)+".cont.taylor.0.restored.components.txt"

    selavypol=path+"selavy-cont-image.i."+base+"SB"+str(sb)+".cont.taylor.0.restored/selavy-image.i."+base+"SB"+str(sb)+".cont.taylor.0.restored.polarisation.txt"

    #metadata
    footfile=path+"metadata/footprintOutput-sb"+str(sb[0:sb.find('.')])+"-"+base+".txt" #'../newfootprint.dat'
    #footfile=path+"metadata/footprintOutput-sb10168-POSSUM_2140-50.txt"

    #mslist file: check if there is more than one (not including the cal one)
    numfiles=0
    for file in os.listdir(path+"metadata/"):
        if(re.match(r'mslist-\d+-\d+-\d+_\d+_\d+.txt$', file)):
            numfiles=numfiles+1
            mslist=path+"metadata/"+file
    if(numfiles==0): print("mslist file not found!")
    elif(numfiles>1): print("More than one mslist file is found!")

    #Oppermann FD map is used to calculate a FD for the field that can be used for
    #comparison
    opDataFile='data/2015_phi_map.fits'
    opErrFile='data/2015_phi_err_map.fits'
    
    files={'iCube':   in_i,\
           'qCube':   in_q,\
           'uCube':   in_u,\
           'vCube':   in_v,\
           'iResCube':   res_i,\
           'qResCube':   res_q,\
           'uResCube':   res_u,\
           'vResCube':   res_v,\
           'iWtCube':   w_i,\
           'qWtCube':   w_q,\
           'uWtCube':   w_u,\
           'vWtCube':   w_v,\
           'i_t0':  taylor0,\
           'i_t1':  taylor1,\
           'iRes_t0':  restaylor0,\
           'iRes_t1':  restaylor1,\
           'iWt_t0':  wtaylor0,\
           'iWt_t1':  wtaylor0,\
           'selavy_comp': selavy,\
           'selavy_pol': selavypol,\
           'footprint': footfile,
           'mslist':    mslist,
           'OpData':    opDataFile,
           'OpErr':     opErrFile\
           }
    return files
    
def verifyFiles(files):
    verified=[]

    for f in files.items():
        if os.path.exists(f[1]):
            verified.append(True)
        else:
            verified.append(False)
            print("Warning: "+f[1]+" does not exist.")
    return verified


def createOutputDirs(path):
    ## Make directories for outputs:
    ## path+/plots
    ## path+/plots/sources
    ## path+/polIm

    if os.path.exists(path+'/plots'):
        print(path +'/plots : exists')
    else:
        os.mkdir(path+'/plots')

    if os.path.exists(path+'/plots/sources'):
        print(path +'/plots/sources : exists')
    else:
        os.mkdir(path+'/plots/sources')

    if os.path.exists(path+'/polIm'):
        print(path +'/polIm : exists')
    else:
        os.mkdir(path+'/polIm')
        
    if os.path.exists(path+'/validation'):
        print(path +'/validation : exists')
    else:
        os.mkdir(path+'/validation')


def mergeSelavy(selavy, selavypol):
    #read selavy input for continuum sources
    df = pd.read_fwf(selavy, header=[0], skiprows=[1], infer_nrows=10000)
    pd.set_option('display.max_colwidth', -1)
    
    df=df[['component_id',\
       'freq',\
       'flux_peak',
       'spectral_index']]

    #read polarized selavy input
    df_pol = pd.read_fwf(selavypol,header=[0],skiprows=[1],infer_nrows=10000)#, skiprows=0,  header=[0,1],index=False)

    #Make a column for polarized fraction (the one that is already in the selavy output
    #only has 2 decimal places - not enough
    df_pol.insert(10,"pol_peak/flux_I_med", df_pol['pol_peak']/df_pol['flux_I_median'])

    df_pol.insert(10,"med_abs_leakage", np.abs(df_pol['flux_V_median']/df_pol['flux_I_median']))


    #Merge polarized and continuum selavy tables
    merged = pd.merge(left=df_pol,right=df, left_on='component_id', right_on='component_id')
    return merged

def makeHoverLink(plotfilename, linktext):
    link="<div class=\"hover_img\"><a href=\'" + plotfilename +"\'>" + linktext + "<span><img src=\"" + plotfilename + "\"> </span></a></div>"
    return link
    
def makeFileLink(plotfilename, linktext):
    link="<a href=\'"+ plotfilename +"'>"+ linktext + "</a>"
    return link

#maybe not needed    
#def addHoverLinkColumn(df, column_name, plotfilename, linktext):
#    #make a new column called "source", which contains the hover code for the html file
#    df[column_name] = makeHoverLinkToPlot(plotfilename, linktext)
#    #df_pol["source"] = "<div class=\"hover_img\"><a href=\'plots/sources/" + df_pol["component_name"].astype(str) + ".png\'>" +df_pol["component_name"].astype(str) + "<span><img src=\"plots/sources/" + df_pol["component_name"].astype(str) + ".png\"> </span></a></div>"


def makeHoverLinkToIQUVim(plotfilename, sb, base):
    i_im=plotfilename+".i.SB"+sb+".contcube."+base+".png"
    q_im=plotfilename+".q.SB"+sb+".contcube."+base+".png"
    u_im=plotfilename+".u.SB"+sb+".contcube."+base+".png"
    v_im=plotfilename+".v.SB"+sb+".contcube."+base+".png"
    i_link="<div class=\"hover_img\"><a href=\'" + i_im +"\'>" + " I " + "<span><img src=\"" + i_im + "\"> </span></a></div>"
    q_link="<div class=\"hover_img\"><a href=\'" + q_im +"\'>" + " Q " + "<span><img src=\"" + q_im + "\"> </span></a></div>"
    u_link="<div class=\"hover_img\"><a href=\'" + u_im +"\'>" + " U " + "<span><img src=\"" + u_im + "\"> </span></a></div>"
    v_link="<div class=\"hover_img\"><a href=\'" + v_im +"\'>" + " V " + "<span><img src=\"" + v_im + "\"> </span></a></div>"

    return [i_link, q_link, u_link, v_link]

    
def filterDataframe(merged, fluxmin=None, fluxmax=None, polmin=None, polmax=None, snrmin=None, snrmax=None):    
    #tot_numsources=merged.shape[0]

    filtered=merged
    if(snrmin!=None): filtered=filtered[filtered.pol_peak_fit_snr > snrmin]
    if(snrmax!=None): filtered=filtered[filtered.pol_peak_fit_snr < snrmax]
    if(fluxmin!=None): filtered=filtered[filtered.flux_peak > fluxmin]
    if(fluxmax!=None): filtered=filtered[filtered.flux_peak < fluxmax]
    if(polmin!=None): filtered=filtered[filtered.pol_peak > polmin]
    if(polmax!=None): filtered=filtered[filtered.pol_peak < polmax]
    
    #filtered=merged[(merged.flux_peak > fluxmin) & (merged.flux_peak < fluxmax)]# & (merged.pol_peak < fluxmax)]

    filtered=filtered.sort_values(by=["flux_peak"], ascending=False)

    filtered.rename(index=str, columns={'flux_peak':'flux_peak (mJy/beam)','flux_I_median':'flux_I_median (mJy/beam)'},inplace=True)
    return filtered

    


def makeHTML(rootpath, sb, base, fluxmin, fluxmax, fieldstats, obsstats, df, doResiduals, doWeights, imageLinks, makeTar):

    html=rootpath+"pol_data_SB"+sb+".html"
    #os.system("rm "+html)
    title="Polarization properties for SB"+str(sb)    
    f=open(html, 'a')
    f.write("<!doctype html><title>"+title+"</title>\n")
    f.write("<link rel=stylesheet type=text/css href='style.css'>\n")
    f.write("<div class=page>\n")
    f.write("<h3>Summary statistics for SB"+str(sb)+"</h3><br>\n")
    f.write("<div class=\"img\"><a href=\"diagnostics/image.i.SB"+sb+".cont.taylor.0.restored_sources.png\"><img style=\"width:600px\" src=\"diagnostics/image.i.SB"+sb+".cont.taylor.0.restored_sources.png\"></a></div>")

    f.close()
    obsstats.to_html(open(html, 'a'), escape=False, header=False, float_format='{:20,.2f}'.format)
    fieldstats.to_html(open(html, 'a'), escape=False,  float_format='{:20,.2f}'.format)
    f=open(html, 'a')
    
    if(makeTar): f.write("<a href=\"plots/fitsim.tar\">Download FITS cutouts for each source (tarball)</a>\n")
    
    f.write("<h3>Summary plots</h3>\n")

    for link in imageLinks:
        f.write(link)    

    f.write("<br>&nbsp;")
    f.close()
 
    plots = pd.DataFrame()
    plots["Beam Noise (image)"] = makeHoverLinkToIQUVim("diagnostics/beamNoise_image.restored", sb, "POSSUM_2140-50")
    plots["Beam Noise (residual)"] = makeHoverLinkToIQUVim("diagnostics/beamNoise_residual", sb, "POSSUM_2140-50")
    plots["Beam Min/Max (image)"] = makeHoverLinkToIQUVim("diagnostics/beamMinMax_image.restored", sb, "POSSUM_2140-50")
    plots["Beam Min/Max (residual)"] = makeHoverLinkToIQUVim("diagnostics/beamMinMax_residual", sb, "POSSUM_2140-50", )
    plots["Beam PSF"] = makeHoverLinkToIQUVim("diagnostics/beamPSF_image.restored", sb, "POSSUM_2140-50")
    plotsT = plots.transpose()
    plotsT.to_html(open(html, 'a'), escape=False, header=False,  float_format='{:20,.2f}'.format)
    f=open(html, 'a')
 
    f.write("<h3>Source properties (sources brighter than "+str(fluxmin)+" mJy)</h3>")
    f.write("</div>")
    f.close()

    df.to_html(open(html, 'a'), escape=False, float_format='{:20,.3f}'.format) #float_format='{:20,.2f}'.format
    
def writeCSS(path):
    file=path+"style.css"
    f=open(file, 'a')
    
    f.write("body            { font-family: \"Lucida Sans Unicode\", \"Lucida Grande\", sans-serif;}\n")
    f.write("a, h1, h2       { color: #377ba8; }\n")
    f.write("h1, h2          { margin: 0; }\n")
    f.write("h1              { border-bottom: 2px solid #eee; }\n")
    f.write("h2              { font-size: 1.2em; }\n")
    f.write("table, th, td {\n")
    f.write("  border: none;\n")
    f.write("  border-bottom: 1px solid #C8C8C8;\n")
    f.write("  border-collapse: collapse;\n")
    f.write("  text-align:left;\n")
    f.write("  padding: 10px;\n")
    f.write("  margin-bottom: 40px;\n")
    f.write("  font-size: 0.9em;\n")
    f.write("  white-space: nowrap;\n")
    f.write("}\n")
    f.write(".hover_img a { position:relative; }\n")
    f.write(".hover_img a span { position:absolute; display:none; z-index:99; left:200px; top: -1000%;  }\n")
    f.write(".hover_img a:hover span {display:block; opacity: 0.95;}\n")
    f.write("img {\n")
    f.write("  float: right;\n")
    f.write("}")
    
    
    f.close()
    return

def make_sbatch(sbatch, account, time, nodes, ntasks, sb, slurmout, pickleout, incube, filtered_csv, imsize, runValidation, makeTar, path):

    f=open(sbatch, 'a')
    f.write("#!/bin/bash\n")
    
    f.write("#SBATCH --account="+account+"\n") # account="rrg-eros-ab"
    f.write("#SBATCH --time="+time+"\n") #time="6:00:00"
    f.write("#SBATCH --nodes="+nodes+"\n")
    f.write("#SBATCH --ntasks="+ntasks+"\n")
    f.write("#SBATCH --job-name=validation-"+sb+"\n")
    f.write("#SBATCH --output=\'"+slurmout+"\'"+"\n") #slurmout="/scratch/b/bmg/westjl/slurmout/slurm-validation-%j.out"
    f.write("module load python/3.6.4-anaconda5.1.0"+"\n")
    f.write("source activate py36"+"\n")
    #f.write("module load gnu-parallel"+"\n")
    #f.write("parallel=\"parallel --delay 0.2 -j $SLURM_NTASKS --joblog runtask.log --resume\""+"\n")
    for i in range(len(pickleout)):
        f.write("srun -n 1 python tabulateSpectra.py "+pickleout[i]+" "+incube[i] +" & \n") # --cpu_bind=cores re.sub('[\[\]]', '', np.array_str(ra[i], precision=2))+" "+re.sub('[\[\]]', '', np.array_str(dec[i],precision=2))
        #f.write("$parallel \"srun -n1 python tabulateSpectra.py "+pickleout[i]+" "+incube[i]+"\"\n")
    #f.write("$parallel \"srun --exclusive -n1 python tabulateSpectra.py "+pickleout[len(pickleout)-1]+" "+incube[len(pickleout)-1]+"\"\n")
    f.write("wait\n")
    if(runValidation):
        f.write("python doValidation.py\n")
    if(makeTar):
        f.write("tar -cf "+path +"/plots/fitsim.tar "+ path + "/polIm")

    f.close()

def make_galaxy_sbatch(sbatch, account, time, nodes, ntasks, sb, slurmout, pickleout, incube, filtered_csv, imsize, runValidation, makeTar, path):

    f=open(sbatch, 'a')
    f.write("#!/bin/bash -l\n")
    
    f.write("#SBATCH --account="+account+"\n") # account="rrg-eros-ab"
    f.write("#SBATCH --time="+time+"\n") #time="6:00:00"
    f.write("#SBATCH --nodes="+nodes+"\n")
    f.write("#SBATCH --ntasks="+ntasks+"\n")
    f.write("#SBATCH --job-name=validation-"+sb+"\n")
    f.write("#SBATCH --output=\'"+slurmout+"\'"+"\n") #slurmout="/scratch/b/bmg/westjl/slurmout/slurm-validation-%j.out"
`   f.write("#SBATCH --export=NONE\n")
	f.write("#SBATCH --clusters=galaxy\n")
	f.write("#SBATCH --partition=workq\n")
	 #for galaxy
    f.write("module load polVal"+"\n")

    for i in range(len(pickleout)):
        f.write("srun --export=ALL --ntasks="+ntasks+" --ntasks-per-node="+ntasks+" python tabulateSpectra.py "+pickleout[i]+" "+incube[i] +" & \n") # --cpu_bind=cores re.sub('[\[\]]', '', np.array_str(ra[i], precision=2))+" "+re.sub('[\[\]]', '', np.array_str(dec[i],precision=2))
        
        #f.write("$parallel \"srun -n1 python tabulateSpectra.py "+pickleout[i]+" "+incube[i]+"\"\n")
    #f.write("$parallel \"srun --exclusive -n1 python tabulateSpectra.py "+pickleout[len(pickleout)-1]+" "+incube[len(pickleout)-1]+"\"\n")
    f.write("wait\n")
    if(runValidation):
        f.write("python doValidation.py\n")
    if(makeTar):
        f.write("tar -cf "+path +"/plots/fitsim.tar "+ path + "/polIm")

    f.close()



def tabulateSpectra(pickleout, incube, filtered_csv, imsize):

    #get source statistics
    filtered = pd.read_csv(filtered_csv)
    ra=filtered["ra_deg_cont"]
    dec=filtered["dec_deg_cont"]
    print("Tabulating spectra for " +str(len(ra)) + " sources in "+ incube) #name='Stokes I'
    print(datetime.datetime.now())
    w = WCS(incube)
    freq=vu.getFrequencies(incube)
    Stats = vu.getSourceStats(incube, ra, dec, w, freq, imsize)
    print("Writing "+pickleout)
    print(datetime.datetime.now())
    #write stats to a file
    with open(pickleout, 'wb') as fp:
        pickle.dump(Stats, fp)
    
    return Stats

