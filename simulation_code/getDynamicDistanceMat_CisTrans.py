#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:13:40 2023

@author: dvirs
"""

import polychrom.polymer_analyses
from polychrom.hdf5_format import list_URIs, load_URI
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool, cpu_count
import seaborn as sb
import glob
from scipy.spatial import distance_matrix
import time
from copy import deepcopy
import matplotlib as mpl
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Analysis of 3D chromosome structure model')
    parser.add_argument('-m', '--model_path', default='240604_ensemble100_normalizedR_50xR',type=str, help="path to model output")
    
    return parser

    
def get_instance_distance(folder, blocks_per_bin, N, copyNum):
    st = time.time()
    instance_num = int(folder[-3:])
    print('getting contacts instance #%d\n' % instance_num)
    URIs = list_URIs(folder)
    URIs = URIs[1:]
    numBins = int(len(URIs)/blocks_per_bin)
    mass_array_bins =  np.loadtxt(folder + "/massArray_bins.csv", delimiter=',')
    
    # distanceMat = []
    # distanceMatReduced = []
    
    cisMat = []
    cisMatReduced = []
    transMat = []
    transMatReduced = []
    
    n=0
    for i in range(numBins):
        # distanceMat.append(np.zeros((N,N)))
        # distanceMatReduced.append(np.zeros((N,N)))
        cisMat.append(np.zeros((N,N)))
        cisMatReduced.append(np.zeros((N,N)))
        transMat.append(np.zeros((N,N)))
        transMatReduced.append(np.zeros((N,N)))
        
        for j in range(blocks_per_bin):
            # if i<29:
            #     n+=1
            #     continue
            U = URIs[n]
            frame = load_URI(U)
            # print("load URI %d accumulated runtime: %f" % (n,time.time() - st))
            positions = frame["pos"]
            positionsReduced = np.zeros((len(positions),2))
            positionsReduced[:,1] = positions[:,2]
            positionsReduced[:,0] = np.sqrt(positions[:,0]**2 + positions[:,1]**2)
            
            distanceMatTmp = distance_matrix(positions, positions)
            distanceMatReducedTmp = distance_matrix(positionsReduced, positionsReduced)
            
            zeroMass_inds = np.where(mass_array_bins[:,i]==0)
            distanceMatTmp[:,zeroMass_inds] = np.nan
            distanceMatTmp[zeroMass_inds,:] = np.nan
            distanceMatReducedTmp[:,zeroMass_inds] = np.nan
            distanceMatReducedTmp[zeroMass_inds,:] = np.nan
            
            replicated_inds_chr2 = np.where(mass_array_bins[N:2*N,i]>0)[0]
            replicated_inds_chr3 = np.where(mass_array_bins[2*N:3*N,i]>0)[0]
            replicated_inds_chr4 = np.where(mass_array_bins[3*N:4*N,i]>0)[0]
            
            cisMatTmp = deepcopy(distanceMatReducedTmp)
            transMatTmp = deepcopy(distanceMatReducedTmp)
            
            
                
            ##Chr1
            indsRow = range(N)
            indsCol = range(3*N,4*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = replicated_inds_chr2
            indsCol = range(N,2*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.intersect1d(replicated_inds_chr2, replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(replicated_inds_chr2, replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(range(N), replicated_inds_chr2)
            indsCol = range(N,2*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.intersect1d(np.setdiff1d(range(N), replicated_inds_chr2), replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(np.setdiff1d(range(N), replicated_inds_chr2), replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            ##Chr2
            indsRow = range(N,2*N)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = replicated_inds_chr2
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N,2*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = np.setdiff1d(range(N),replicated_inds_chr2)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.intersect1d(range(N,2*N), replicated_inds_chr4+N)
            indsCol = range(3*N,4*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(range(N,2*N), replicated_inds_chr4+N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
           
                
            ##Chr3
            indsRow = range(2*N,3*N)
            indsCol = range(3*N,4*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N,2*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = replicated_inds_chr3
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(2*N,3*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = np.setdiff1d(range(N),replicated_inds_chr3)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            ##Chr4
            indsRow = range(3*N,4*N)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = replicated_inds_chr4+N
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(3*N,4*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = np.setdiff1d(range(N,2*N),replicated_inds_chr4+N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            cisMatTmpReduced = deepcopy(cisMatTmp)
            transMatTmpReduced = deepcopy(transMatTmp)
            
            cisMatTmp = deepcopy(distanceMatTmp)
            transMatTmp = deepcopy(distanceMatTmp)
            
            ##Chr1
            indsRow = range(N)
            indsCol = range(3*N,4*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = replicated_inds_chr2
            indsCol = range(N,2*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.intersect1d(replicated_inds_chr2, replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(replicated_inds_chr2, replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(range(N), replicated_inds_chr2)
            indsCol = range(N,2*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.intersect1d(np.setdiff1d(range(N), replicated_inds_chr2), replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(np.setdiff1d(range(N), replicated_inds_chr2), replicated_inds_chr3)
            indsCol = range(2*N,3*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            ##Chr2
            indsRow = range(N,2*N)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = replicated_inds_chr2
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N,2*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = np.setdiff1d(range(N),replicated_inds_chr2)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.intersect1d(range(N,2*N), replicated_inds_chr4+N)
            indsCol = range(3*N,4*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsRow = np.setdiff1d(range(N,2*N), replicated_inds_chr4+N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            ##Chr3
            indsRow = range(2*N,3*N)
            indsCol = range(3*N,4*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N,2*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = replicated_inds_chr3
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(2*N,3*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = np.setdiff1d(range(N),replicated_inds_chr3)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            ##Chr4
            indsRow = range(3*N,4*N)
            indsCol = range(2*N,3*N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(N)
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = replicated_inds_chr4+N
            cisMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            cisMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = range(3*N,4*N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            indsCol = np.setdiff1d(range(N,2*N),replicated_inds_chr4+N)
            transMatTmp[np.ix_(indsRow, indsCol)] = np.nan
            transMatTmp[np.ix_(indsCol, indsRow)] = np.nan
            
            for j in range(N):
                tmpCis = np.zeros((copyNum,copyNum*N))
                tmpCisR = np.zeros((copyNum,copyNum*N))
                tmpTrans = np.zeros((copyNum,copyNum*N))
                tmpTransR = np.zeros((copyNum,copyNum*N))
                for k in range(copyNum):
                    tmpCis[k,:] = cisMatTmp[k*N+j,:]
                    tmpCisR[k,:] = cisMatTmpReduced[k*N+j,:]
                    tmpTrans[k,:] = transMatTmp[k*N+j,:]
                    tmpTransR[k,:] = transMatTmpReduced[k*N+j,:]
                tmpCis = tmpCis.reshape((copyNum*copyNum,N))
                tmpCisR = tmpCisR.reshape((copyNum*copyNum,N))
                tmpTrans = tmpTrans.reshape((copyNum*copyNum,N))
                tmpTransR = tmpTransR.reshape((copyNum*copyNum,N))
                
                tmp = np.nanmin(tmpCis[:,:N],axis=0)
                tmp[np.isnan(tmp)]=0
                cisMat[i][:,j] += tmp
                tmp = np.nanmin(tmpCisR[:,:N],axis=0)
                tmp[np.isnan(tmp)]=0
                cisMatReduced[i][:,j] += tmp
                tmp = np.nanmin(tmpTrans[:,:N],axis=0)
                tmp[np.isnan(tmp)]=0
                transMat[i][:,j] += tmp
                tmp = np.nanmin(tmpTransR[:,:N],axis=0)
                tmp[np.isnan(tmp)]=0
                transMatReduced[i][:,j] += tmp
            
            # if i==29:
            #     plt.imshow(cisMatTmp)
            #     plt.savefig('debug.png')
            #     plt.close()
            #     print(replicated_inds_chr2)
            #     print(replicated_inds_chr3)
            #     print(replicated_inds_chr4)
            #     exit(0)
           
                
            n+=1
    return cisMat, cisMatReduced, transMat, transMatReduced

parser=create_parser()
args = parser.parse_args()

modelsPath = args.model_path

N_MG1655 = 4641652
copyNum = 4
blocks_per_bin = 10
# modelsPath = "full_cell_cycle_model_200k_2D_pooledExp_angK4_multiInstancesGPU_230719" 
# modelsPath = "full_cell_cycle_model_200k_2D_pooledExp_multiInstancesGPU_repInput_angK4_weakTether_230802"
# modelsPath = "full_cell_cycle_model_200k_2D_pooledExp_multiInstancesGPU_repInput_angK2_weakTether_N1k_230815"
# modelsPath = "full_cell_cycle_model_2D_pooledExp_multiInstancesGPU_repInput_angK2_N1k_noRestraints_230825"
# modelsPath = "240604_ensemble100_normalizedR_50xR"
# modelsPath = "240502_ensemble100_noRestraints"

# modelsPath = "full_cell_cycle_model_2D_pooledExp_multiInstancesGPU_repInput_angK2_N1k_230822"
modelsList = glob.glob(modelsPath + '/*')
modelsList.sort()
rm_list = []
for i, m in enumerate(modelsList):
    try:
        list_URIs(m)
    except:
        # list_models.remove(m)
        rm_list.append(i)
# list_models = list_models[:2]
rm_list.sort(reverse=True)
for i in rm_list:
    modelsList.pop(i)    

URIs = list_URIs(modelsList[0])
numBins = int(len(URIs)/blocks_per_bin)
first_frame = load_URI(URIs[0])
# N = int(len(first_frame["pos"])/2)
N = int(len(first_frame["pos"])/copyNum)

MD = np.array([3759738, 4641652])
MD = MD/N_MG1655*N
oriMD = MD.astype('int')
MD = np.array([603414, 1206829])
MD = MD/N_MG1655*N
rightMD = MD.astype('int')
MD = np.array([2181576+1, 2877824])
MD = MD/N_MG1655*N
leftMD = MD.astype('int')
MD = np.array([1206829+1, 2181576])
MD = MD/N_MG1655*N
terMD = MD.astype('int')

mass_array_bins =  np.loadtxt(modelsList[0] + "/massArray_bins.csv", delimiter=',')

# localization_restraints_file = '../data/230615_localization_data_radial.csv'
localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
restraints = pd.read_csv(localization_restraints_file,sep = ',')

volume_ratio = 0.1
r_monomer = 0.5 #never change
v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
R=L/2 #radius of cylinder
L = round(L,1)
R = round(R,1)
Lf = ((restraints.cell_length.values[-1]-restraints.cell_length.values[0])/restraints.cell_length.values[0]+1)*L

# modelsList = modelsList[:16]
numModels = len(modelsList)

args = []
for m in modelsList:
    args.append((m,blocks_per_bin,N, copyNum))

# [cis, cisR, trans, transR] = get_instance_distance(modelsList[0],blocks_per_bin,N,copyNum)
# print(contactMap.toarray())
# print('************')
# print(contactMapReduced.toarray())

with Pool(cpu_count()) as p:
    Results = p.starmap(get_instance_distance, args)
# print(Results)
print('combining contacts\n')
dynamicCisMat = []
dynamicCisMatReduced = []
dynamicTransMat = []
dynamicTransMatReduced = []
for i in range(numBins):
    for r in Results:
        dynamicCisMat.append(r[0][i])
        dynamicCisMatReduced.append(r[1][i])
        dynamicTransMat.append(r[2][i])
        dynamicTransMatReduced.append(r[3][i])


print('plotting contacts\n')

outFolder = "dynamicDistanceMat_CisTrans_%s" % modelsPath
# outFolder = "dynamicHiC_angK2_N1K"
if not os.path.isdir(outFolder):
    os.mkdir(outFolder)
    os.mkdir(outFolder + '/3DContacts')
    os.mkdir(outFolder + '/ReducedContacts')

oriC_site = 3925860
terBead = int((oriC_site-N_MG1655/2)/N_MG1655*N)
oriBead = int(oriC_site/N_MG1655*N)

cmap = mpl.cm.get_cmap("vlag_r").copy()
cmap.set_bad('0')

window_size = 1
thr = []
thrReduced = []
thrL = []
thrReducedL = []
thrT = []
thrTReduced = []
thrTL = []
thrTReducedL = []
combinedCisMat = np.zeros((numBins,N,N))
combinedCisMatReduced = np.zeros((numBins,N,N))
combinedTransMat = np.zeros((numBins,N,N))
combinedTransMatReduced = np.zeros((numBins,N,N))
for i in range(int(np.floor(window_size/2)), numBins-int(np.floor(window_size/2))):
    # combinedContactMap[i] = np.zeros((N,N))
    # combinedContactMapReduced[i] = np.zeros((N,N))
    for j in range(-int(np.floor(window_size/2)*numModels),(int(np.floor(window_size/2))+1)*numModels):
        # print(numModels*i+j)
        combinedCisMat[i] += dynamicCisMat[numModels*i+j]
        combinedCisMatReduced[i] += dynamicCisMatReduced[numModels*i+j]
        combinedTransMat[i] += dynamicTransMat[numModels*i+j]
        combinedTransMatReduced[i] += dynamicTransMatReduced[numModels*i+j]
    combinedCisMat[i][combinedCisMat[i]==0]=np.nan
    combinedCisMat[i][range(N),range(N)]=0
    combinedCisMatReduced[i][combinedCisMatReduced[i]==0]=np.nan
    combinedCisMatReduced[i][range(N),range(N)]=0
    combinedTransMat[i][combinedTransMat[i]==0]=np.nan
    combinedTransMatReduced[i][combinedTransMatReduced[i]==0]=np.nan
    
    beadNum = np.sum(mass_array_bins[:,i]>0)
    combinedCisMat[i] = combinedCisMat[i]/(blocks_per_bin*len(modelsList)*window_size)
    combinedCisMatReduced[i] = combinedCisMatReduced[i]/(len(URIs)*len(modelsList)*window_size)
    thr.append(np.quantile(combinedCisMat[i],0.80))
    thrReduced.append(np.quantile(combinedCisMatReduced[i],0.80))
    thrL.append(np.quantile(combinedCisMat[i],0.025))
    thrReducedL.append(np.quantile(combinedCisMatReduced[i],0.025))
    combinedTransMat[i] = combinedTransMat[i]/(blocks_per_bin*len(modelsList)*window_size)
    combinedTransMatReduced[i] = combinedTransMatReduced[i]/(len(URIs)*len(modelsList)*window_size)
    thrT.append(np.quantile(combinedTransMat[i],0.80))
    thrTReduced.append(np.quantile(combinedTransMatReduced[i],0.80))
    thrTL.append(np.quantile(combinedTransMat[i],0.025))
    thrTReducedL.append(np.quantile(combinedTransMatReduced[i],0.025))
    
np.save('simDistMatCisDynamic_%s.npy' % modelsPath, combinedCisMat)
np.save('simDistMatReducedDimCisDynamic_%s.npy' % modelsPath, combinedCisMatReduced)
np.save('simDistMatTransDynamic_%s.npy' % modelsPath, combinedTransMat)
np.save('simDistMatReducedDimTransDynamic_%s.npy' % modelsPath, combinedTransMatReduced)

thr = np.mean(np.array(thr))
thrReduced = np.mean(np.array(thrReduced))
thrL = np.mean(np.array(thrL))
thrReducedL = np.mean(np.array(thrReducedL))
thrT = np.mean(np.array(thrT))
thrTReduced = np.mean(np.array(thrTReduced))
thrTL = np.mean(np.array(thrTL))
thrTReducedL = np.mean(np.array(thrTReducedL))
for i in range(int(np.floor(window_size/2)), numBins-int(np.floor(window_size/2))):
    beadsToGenome = list(map(int,np.linspace(0,N_MG1655/1e3,N)))
    
    CisMatdf = pd.DataFrame(combinedCisMat[i],columns=beadsToGenome,index=beadsToGenome)
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(CisMatdf,vmin=thrL,vmax=thr,cmap=cmap,ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='g')
    ax1.axvline(x=oriBead, linewidth=1, color='orange')
    ax1.axhline(y=0, xmin=oriMD[0]/N, xmax=oriMD[1]/N, color='orange',linewidth=5)
    ax1.axhline(y=0, xmin=terMD[0]/N, xmax=terMD[1]/N, color='g',linewidth=5)
    ax1.axhline(y=0, xmin=rightMD[0]/N, xmax=rightMD[1]/N, color='m',linewidth=5)
    ax1.axhline(y=0, xmin=leftMD[0]/N, xmax=leftMD[1]/N, color='purple',linewidth=5)
    plt.savefig('%s/3DContacts/simDynamicCisMat%d.png' % (outFolder, i))
    # plt.show()
    plt.close(fig)
    
    CisMatReduceddf = pd.DataFrame(combinedCisMatReduced[i],columns=beadsToGenome,index=beadsToGenome)
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(CisMatReduceddf,vmin=thrReducedL,vmax=thrReduced,cmap=cmap,ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='g')
    ax1.axvline(x=oriBead, linewidth=1, color='orange')
    ax1.axhline(y=0, xmin=oriMD[0]/N, xmax=oriMD[1]/N, color='orange',linewidth=5)
    ax1.axhline(y=0, xmin=terMD[0]/N, xmax=terMD[1]/N, color='g',linewidth=5)
    ax1.axhline(y=0, xmin=rightMD[0]/N, xmax=rightMD[1]/N, color='m',linewidth=5)
    ax1.axhline(y=0, xmin=leftMD[0]/N, xmax=leftMD[1]/N, color='purple',linewidth=5)
    plt.savefig('%s/ReducedContacts/simDynamicCisMatReduced%d.png' % (outFolder, i))
    # plt.show()
    plt.close(fig)
    
    TransMatdf = pd.DataFrame(combinedTransMat[i],columns=beadsToGenome,index=beadsToGenome)
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(TransMatdf,vmin=thrL,vmax=thr,cmap=cmap,ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='g')
    ax1.axvline(x=oriBead, linewidth=1, color='orange')
    ax1.axhline(y=0, xmin=oriMD[0]/N, xmax=oriMD[1]/N, color='orange',linewidth=5)
    ax1.axhline(y=0, xmin=terMD[0]/N, xmax=terMD[1]/N, color='g',linewidth=5)
    ax1.axhline(y=0, xmin=rightMD[0]/N, xmax=rightMD[1]/N, color='m',linewidth=5)
    ax1.axhline(y=0, xmin=leftMD[0]/N, xmax=leftMD[1]/N, color='purple',linewidth=5)
    plt.savefig('%s/3DContacts/simDynamicTransMat%d.png' % (outFolder, i))
    # plt.show()
    plt.close(fig)
    
    TransMatReduceddf = pd.DataFrame(combinedTransMatReduced[i],columns=beadsToGenome,index=beadsToGenome)
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(TransMatReduceddf,vmin=thrReducedL,vmax=thrReduced,cmap=cmap,ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='g')
    ax1.axvline(x=oriBead, linewidth=1, color='orange')
    ax1.axhline(y=0, xmin=oriMD[0]/N, xmax=oriMD[1]/N, color='orange',linewidth=5)
    ax1.axhline(y=0, xmin=terMD[0]/N, xmax=terMD[1]/N, color='g',linewidth=5)
    ax1.axhline(y=0, xmin=rightMD[0]/N, xmax=rightMD[1]/N, color='m',linewidth=5)
    ax1.axhline(y=0, xmin=leftMD[0]/N, xmax=leftMD[1]/N, color='purple',linewidth=5)
    plt.savefig('%s/ReducedContacts/simDynamicTransMatReduced%d.png' % (outFolder, i))
    # plt.show()
    plt.close(fig)
    

