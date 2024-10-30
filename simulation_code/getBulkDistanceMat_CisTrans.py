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
from copy import deepcopy
import time
import matplotlib as mpl
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Analysis of 3D chromosome structure model')
    parser.add_argument('-m', '--model_path', default='240604_ensemble100_normalizedR_50xR',type=str, help="path to model output")
    
    return parser

    
def get_instance_distanceMat(folder, blocks_per_bin, N, copyNum):
    st = time.time()
    instance_num = int(folder[-3:])
    print('getting contacts instance #%d\n' % instance_num)
    URIs = list_URIs(folder)
    URIs = URIs[1:]
   
    mass_array_bins =  np.loadtxt(folder + "/massArray_bins.csv", delimiter=',')
    
    distanceMatTmp = np.zeros((copyNum*N,copyNum*N))
 
    cisMat = np.zeros((N,N))
    transMat = np.zeros((N,N))
    for k,U in enumerate(URIs):
        # print("%d/%d" % (i, len(URIs)))
        frame = load_URI(U)
        # print("load URI %d accumulated runtime: %f" % (k,time.time() - st))
        positions = frame["pos"]
        
        distanceMatTmp = distance_matrix(positions, positions)
     
        zeroMass_inds = np.where(mass_array_bins[:,int(np.floor(k/blocks_per_bin))]==0)
        distanceMatTmp[:,zeroMass_inds] = np.nan
        distanceMatTmp[zeroMass_inds,:] = np.nan
        
        replicated_inds_chr2 = np.where(mass_array_bins[N:2*N,int(np.floor(k/blocks_per_bin))]>0)[0]
        replicated_inds_chr3 = np.where(mass_array_bins[2*N:3*N,int(np.floor(k/blocks_per_bin))]>0)[0]
        replicated_inds_chr4 = np.where(mass_array_bins[3*N:4*N,int(np.floor(k/blocks_per_bin))]>0)[0]

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
        
        # print("segement cis & trans mats %d accumulated runtime: %f" % (k,time.time() - st))
                 
        for j in range(N):
            tmpCis = np.zeros((copyNum,copyNum*N))
            tmpTrans = np.zeros((copyNum,copyNum*N))
            for k in range(copyNum):
                tmpCis[k,:] = cisMatTmp[k*N+j,:]
                tmpTrans[k,:] = transMatTmp[k*N+j,:]
            tmpCis = tmpCis.reshape((copyNum*copyNum,N))
            tmpTrans = tmpTrans.reshape((copyNum*copyNum,N))
            
            tmp = np.nanmin(tmpCis[:,:N],axis=0)
            tmp[np.isnan(tmp)]=0
            cisMat[:,j] =  cisMat[:,j] + tmp
            tmp = np.nanmin(tmpTrans[:,:N],axis=0)
            tmp[np.isnan(tmp)]=0
            transMat[:,j] =  transMat[:,j] + tmp
         
    cisMat = cisMat/len(URIs)
    transMat = transMat/len(URIs)
    return cisMat, transMat

parser=create_parser()
args = parser.parse_args()

modelsPath = args.model_path

N_MG1655 = 4641652
copyNum = 4

modelsList = glob.glob(modelsPath + '/*')
modelsList.sort()
rm_list = []
for i, m in enumerate(modelsList):
    try:
        list_URIs(m)
    except:
        rm_list.append(i)

rm_list.sort(reverse=True)
for i in rm_list:
    modelsList.pop(i)  
# modelsList = modelsList[:15]

URIs = list_URIs(modelsList[0])
first_frame = load_URI(URIs[0])
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

localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
restraints = pd.read_csv(localization_restraints_file,sep = ',')

volume_ratio = 0.05
r_monomer = 0.5 #never change
v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
R=L/2 #radius of cylinder
L = round(L,1)
R = round(R,1)
Lf = ((restraints.cell_length.values[-1]-restraints.cell_length.values[0])/restraints.cell_length.values[0]+1)*L

blocks_per_bin = 10

args = []
for m in modelsList:
    args.append((m,blocks_per_bin,N,copyNum))

with Pool(cpu_count()) as p:
    Results = p.starmap(get_instance_distanceMat, args)

print('combining contacts\n')
combinedCisMat = np.zeros((N,N))
combinedTransMat = np.zeros((N,N))
for r in Results:
    combinedCisMat += r[0]
    combinedTransMat += r[1]
combinedCisMat = combinedCisMat/len(modelsList)
combinedTransMat = combinedTransMat/len(modelsList)

combinedCisMat[combinedCisMat==0] = np.nan
combinedCisMat[range(N),range(N)] = 0
combinedTransMat[combinedTransMat==0] = np.nan

if '/' in modelsPath:
    tmp = modelsPath.split('/')
    modelsName = tmp[-1]
else:
    modelsName = modelsPath

np.save('Distance_maps/Data/simDistMatCis_%s.npy' % modelsName, combinedCisMat)
np.save('Distance_maps/Data/simDistMatTrans_%s.npy' % modelsName, combinedTransMat)
