#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 15:01:19 2023

@author: dvirs
"""


from polychrom.hdf5_format import list_URIs, load_URI
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
import seaborn as sb
import glob
import argparse
from time import sleep
from scipy.io import savemat

def create_parser():
    parser = argparse.ArgumentParser(description='Analysis of 3D chromosome structure model')
    parser.add_argument('-m', '--model_path', default='241022_4Paper_normalModel',type=str, help="path to model output")
    
    return parser

def getLocusCoordinates(i, strainCoordinates, blocks_per_bin, modelPath, strainBeadNum, copyNum):
    # print(i)
    URIs = list_URIs(modelPath)
    URIs = URIs[1:]
    mass_array_bins =  np.loadtxt(modelPath + "/massArray_bins.csv", delimiter=',')
    long_axis_c = np.empty((len(URIs),copyNum))
    r = np.empty((len(URIs),copyNum))
    sCoord = np.empty((len(URIs),copyNum))
    dCoord = np.empty((len(URIs),copyNum))
    for j,U in enumerate(URIs):
        frame = load_URI(U)
        long_axis_c[j,:] = frame["pos"][strainBeadNum,2]
        long_axis_c[j,np.where(mass_array_bins[strainBeadNum,int(np.floor(j/blocks_per_bin))]==0)]=np.nan
        x = frame["pos"][strainBeadNum,0]
        x[np.where(mass_array_bins[strainBeadNum,int(np.floor(j/blocks_per_bin))]==0)]=np.nan
        y = frame["pos"][strainBeadNum,1]
        y[np.where(mass_array_bins[strainBeadNum,int(np.floor(j/blocks_per_bin))]==0)]=np.nan
        r[j,:] = np.sqrt(y**2 + x**2)
        sCoord[j,:] = x
        dCoord[j,:] = y
    return long_axis_c, r, sCoord, dCoord

def saveStrainForkPlots(strainID,isrestraint, distToRestraint, modelsPath):
    sleep(strainID*0.1)
    print("Plotting Forkplots strainID %02d" % strainID)
    
    if '/' in modelsPath:
        tmp = modelsPath.split('/')
        modelsName = tmp[-1]
    else:
        modelsName = modelsPath
        
    outFolder = 'simulated_distribution_plots'
    forkPlotsPath = '%s/simulatedForkPlots_%s/' % (outFolder, modelsName)
    radialPlotsPath = '%s/simulatedRadialPlots_%s/' % (outFolder, modelsName)
    doghnutPlotsPath = '%s/simulatedDoughnutPlots_%s/' % (outFolder, modelsName)
    if not os.path.isdir(forkPlotsPath):
        os.mkdir(forkPlotsPath)
        os.mkdir(forkPlotsPath + 'restrainedStrains/')
        os.mkdir(forkPlotsPath + 'nonRestrainedStrains/')
    if not os.path.isdir(radialPlotsPath):
        os.mkdir(radialPlotsPath)
        os.mkdir(radialPlotsPath + 'restrainedStrains/')
        os.mkdir(radialPlotsPath + 'nonRestrainedStrains/')
    if not os.path.isdir(doghnutPlotsPath):
        os.mkdir(doghnutPlotsPath)
        os.mkdir(doghnutPlotsPath + 'restrainedStrains/')
        os.mkdir(doghnutPlotsPath + 'nonRestrainedStrains/')
    
    N_MG1655 = 4641652
    copyNum = 4
    
    localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
    lib_tbl_file = "../data/220518_chromosome_structure_library_sites_BC.csv" 
    restraints = pd.read_csv(localization_restraints_file,sep = ',')
    lib_tbl = pd.read_csv(lib_tbl_file,sep = ',')
    
    list_models = glob.glob(modelsPath + '/*')
    list_models.sort()
    rm_list = []
    for i, m in enumerate(list_models):
        try:
            list_URIs(m)
        except:
            # list_models.remove(m)
            rm_list.append(i)
    # list_models = list_models[:2]
    rm_list.sort(reverse=True)
    for i in rm_list:
        list_models.pop(i)
    
    num_of_models = len(list_models)
    URIs = list_URIs(list_models[0])
    first_frame = load_URI(URIs[0])
    N = int(len(first_frame["pos"])/copyNum)
    
    volume_ratio = 0.05
    r_monomer = 0.5 #never change
    v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
    v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
    L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
    R=L/2 #radius of cylinder
    L = round(L,1)
    R = round(R,1)
    # Lf = ((restraints.cell_length.values[-1]-restraints.cell_length.values[0])/restraints.cell_length.values[0]+1)*L
    
    scaling_factor = L/0.9375
    Lf = 1.6*scaling_factor
    
    num_of_bins = len(restraints)
    blocks_per_bin = int(len(URIs)/num_of_bins)
    genome_loci = restraints.columns[1:]
    genome_loci = [int(a.split(sep='_')[0]) for a in genome_loci]
    genome_loci = np.unique(genome_loci)
    
    genome_loci = lib_tbl.MG1655_ref_genome_location.values
    grid = np.linspace(1,N_MG1655,N,dtype=int)
    bead_num = np.array([np.argmin(abs(grid-i)) for i in genome_loci])
    # mass_array_bins =  np.loadtxt(pathToURIs + "massArray_bins.csv", delimiter=',')
    
    strainLoci = lib_tbl[lib_tbl['id']==strainID]['MG1655_ref_genome_location'].values
    strainBeadNum = bead_num[np.where(genome_loci==strainLoci[0])]
    strainBeadNum = np.append(strainBeadNum, strainBeadNum[0] + N)
    if copyNum==4:
        strainBeadNum = np.append(strainBeadNum, strainBeadNum[0] + 2*N)
        strainBeadNum = np.append(strainBeadNum, strainBeadNum[0] + 3*N)
    strainCoordinates = np.nan*np.ones((num_of_models*copyNum,num_of_bins*blocks_per_bin))
    strainCoordinatesR = np.nan*np.ones((num_of_models*copyNum,num_of_bins*blocks_per_bin))
    strainCoordinatesX = np.nan*np.ones((num_of_models*copyNum,num_of_bins*blocks_per_bin))
    strainCoordinatesY = np.nan*np.ones((num_of_models*copyNum,num_of_bins*blocks_per_bin))

    for i,f in enumerate(list_models):
        long_axis_coords, radial_coords, x, y =  getLocusCoordinates(i, strainCoordinates, blocks_per_bin, f, strainBeadNum, copyNum)
        strainCoordinates[np.arange(copyNum*i,copyNum*(i+1)),:] = np.transpose(long_axis_coords)
        strainCoordinatesR[np.arange(copyNum*i,copyNum*(i+1)),:] = np.transpose(radial_coords)
        
    lBinsNum = 50
    rBinsNum = 50
    xBinsNum = 30
    lBins = np.linspace(-Lf/2, Lf/2, lBinsNum+1)
    rBins = np.linspace(0, R, rBinsNum+1)
    xBins = np.linspace(-R, R, xBinsNum+1)
    yBins = np.linspace(-R, R, xBinsNum+1)
    heatMap_tmp = np.empty((np.shape(strainCoordinates)[1],lBinsNum))
    heatMapR = np.empty((np.shape(strainCoordinatesR)[1],rBinsNum))
    for i,s in enumerate(np.transpose(strainCoordinates)):
        heatMap_tmp[i,:] = np.histogram(s[~np.isnan(s)],lBins,density=True)[0]
    
    dr = np.diff(rBins);
    dr = dr[0];
    for i,s in enumerate(np.transpose(strainCoordinatesR)):
        hr = np.histogram(s[~np.isnan(s)],rBins,density=False)[0]
        hr = hr-1  
        hr[hr<0] = 0;
        hr = hr/np.sum(hr)
        r1 = rBins[:-1]
        r2 = rBins[1:]
        norm_vec = 1/(np.pi*(r2**2-r1**2))
        heatMapR[i,:] = hr*norm_vec
        a=1
            
    heatMapDoughnut = np.histogram2d(x[~np.isnan(x)], y[~np.isnan(y)],[xBins,yBins], density=True)
    #%%    
    lenConversion = restraints.cell_length.values[0]/L
    plt.figure()
    binLenVec = restraints.cell_length.values
    f = sp.interpolate.interp1d(np.arange(0,np.shape(heatMap_tmp)[0]+1,blocks_per_bin), np.append(binLenVec, binLenVec[-1]),kind='zero')
    binLenVecInter = f(range(np.shape(heatMap_tmp)[0]))
    heatMapdf = pd.DataFrame(heatMap_tmp,columns=np.round(lBins[:-1]*lenConversion,1))
    ax = sb.heatmap(heatMapdf, xticklabels=4,yticklabels=20,cmap='jet')
    tmp = ax.get_yticks()
    ax.set_yticks(np.linspace(tmp[0],tmp[-1],int(len(binLenVec)/2)))
    ax.set_yticklabels(np.round(binLenVec[:-1:2],1))
    x=list(map(lambda x: np.argmin(abs(x-lBins)),-binLenVecInter/2/lenConversion))
    plt.plot(x, np.arange(np.shape(heatMap_tmp)[0]),color='w')
    x=list(map(lambda x: np.argmin(abs(x-lBins)),binLenVecInter/2/lenConversion))
    plt.plot(x, np.arange(np.shape(heatMap_tmp)[0]),color='w')
    # plt.plot(x, binLenVecInter,color='w',linewidth=30)
    plt.xlabel('Long axis distribution [$\mu$m]')
    plt.ylabel('Cell length [$\mu$m]')
    plt.title('simulated forkplot strainID %d' % strainID)
    # print(isrestraint)
    if isrestraint:
        # print(forkPlotsPath + 'restrainedStrains/' + 'simForkPLot_M%02d.png' % strainID )
        plt.savefig(forkPlotsPath + 'restrainedStrains/' + 'simForkPLot_M%02d.png' % strainID  )
        savemat(forkPlotsPath + 'restrainedStrains/' + 'simForkPLot_M%02d.mat' % strainID, {'heatMap': heatMapdf.to_numpy()}  )
    else:
        # print(forkPlotsPath + 'nonRestrainedStrains/' + 'simForkPLot_M%02d.png' % strainID)
        plt.title('simulated forkplot strainID %d\ndistance to nearest restraint locus = %d' % (strainID, distToRestraint))
        plt.savefig(forkPlotsPath + 'nonRestrainedStrains/' + 'simForkPLot_M%02d.png' % strainID  )
        savemat(forkPlotsPath + 'nonRestrainedStrains/' + 'simForkPLot_M%02d.mat' % strainID, {'heatMap': heatMapdf.to_numpy()}  )
    #%%
    
    plt.figure()
    heatMapRdf = pd.DataFrame(heatMapR,columns=np.round(rBins[:-1]/R,1))
    ax = sb.heatmap(heatMapRdf, xticklabels=4,yticklabels=20, cmap='jet')
    tmp = ax.get_yticks()
    ax.set_yticks(np.linspace(tmp[0],tmp[-1],int(len(binLenVec)/2)))
    ax.set_yticklabels(np.round(binLenVec[:-1:2],1))
    plt.xlabel('Normalized radial coordinates distribution')
    plt.ylabel('Cell length [$\mu$m]')
    plt.title('simulated radial distribution strainID %d' % strainID)
    if isrestraint:
        plt.savefig(radialPlotsPath + 'restrainedStrains/' +'simRadialDist_M%02d.png' % strainID  )
        savemat(radialPlotsPath + 'restrainedStrains/' +'simRadialDist_M%02d.mat' % strainID,  {'heatMap': heatMapRdf.to_numpy()} )
        
    else:
        plt.title('simulated radial distribution strainID %d\ndistance to nearest restraint locus = %d' % (strainID, distToRestraint))
        plt.savefig(radialPlotsPath + 'nonRestrainedStrains/' +'simRadialDist_M%02d.png' % strainID  )
        savemat(radialPlotsPath + 'nonRestrainedStrains/' +'simRadialDist_M%02d.mat' % strainID,  {'heatMap': heatMapRdf.to_numpy()} )
    #%%
    
    plt.figure()
    heatMapDdf = pd.DataFrame(heatMapDoughnut[0],columns=np.round(yBins[:-1]/R,1))
    ax = sb.heatmap(heatMapDdf, xticklabels=4, yticklabels=4, cmap='jet')
    tick = ax.get_xticklabels()
    ax.set_yticklabels(tick)
    plt.xlabel('Y-axis [$\mu$m]')
    plt.ylabel('Z-axis [$\mu$m]')
    plt.title('simulated YZ distribution strainID %d' % strainID)
    if isrestraint:
        plt.savefig(doghnutPlotsPath + 'restrainedStrains/' +'simYZDist_M%02d.png' % strainID  )
        savemat(doghnutPlotsPath + 'restrainedStrains/' +'simYZDist_M%02d.mat' % strainID, {'heatMap': heatMapDdf.to_numpy()})
    else:
        plt.title('simulated YZ distribution strainID %d\ndistance to nearest restraint locus = %d' % (strainID, distToRestraint))
        plt.savefig(doghnutPlotsPath + 'nonRestrainedStrains/' +'simYZDist_M%02d.png' % strainID  )
        savemat(doghnutPlotsPath + 'nonRestrainedStrains/' +'simYZDist_M%02d.mat' % strainID, {'heatMap': heatMapDdf.to_numpy()} )
        

parser=create_parser()
args = parser.parse_args()

modelsPath = args.model_path

localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
lib_tbl_file = "../data/220518_chromosome_structure_library_sites_BC.csv" 
restraints = pd.read_csv(localization_restraints_file,sep = ',')
lib_tbl = pd.read_csv(lib_tbl_file,sep = ',')

genome_loci = restraints.columns[1:]
genome_loci = [int(a.split(sep='_')[0]) for a in genome_loci]
genome_loci = np.unique(genome_loci)
a, inds1, inds2 = np.intersect1d(lib_tbl.MG1655_ref_genome_location,genome_loci, return_indices=True)
isInRestraints = np.zeros((len(lib_tbl)))
isInRestraints[inds1] = 1
isInRestraints = list(map(bool,isInRestraints))
distToRestraint = np.empty(len(lib_tbl))
N_MG1655 = 4641652
genomic_loci_tmp = lib_tbl.MG1655_ref_genome_location.values[isInRestraints]

for i in range(len(lib_tbl)):
    if not isInRestraints[i]:
        distL = lib_tbl.MG1655_ref_genome_location.values[i]-genomic_loci_tmp[genomic_loci_tmp < lib_tbl.MG1655_ref_genome_location.values[i]]
        if len(distL)==0:
            distL = N_MG1655-np.max(genomic_loci_tmp)+lib_tbl.MG1655_ref_genome_location.values[i]
        else:
            distL = distL[-1]
        distH = genomic_loci_tmp[genomic_loci_tmp > lib_tbl.MG1655_ref_genome_location.values[i]]-lib_tbl.MG1655_ref_genome_location.values[i]
        if len(distH)==0:
            distH = N_MG1655 - lib_tbl.MG1655_ref_genome_location.values[i] + np.min(genomic_loci_tmp)
        else:
            distH = distH[0]
        distToRestraint[i] = min([distL, distH])
        
args = []
for i in range(len(lib_tbl)):
    # saveStrainForkPlots(lib_tbl.id.values[i], isInRestraints[i], distToRestraint[i])
    args.append((lib_tbl.id.values[i], isInRestraints[i], distToRestraint[i], modelsPath))
    
p = Pool(cpu_count())
p.starmap(saveStrainForkPlots, args)
