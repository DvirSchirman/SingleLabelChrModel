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
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Analysis of 3D chromosome structure model')
    parser.add_argument('-m', '--model_path', default='Ensemble_models/241022_4Paper_normalModel',type=str, help="path to model output")
    
    return parser
    
def get_instance_distanceMat(folder, blocks_per_bin, N, copyNum):
    st = time.time()
    instance_num = int(folder[-3:])
    print('getting contacts instance #%d\n' % instance_num)
    URIs = list_URIs(folder)
    URIs = URIs[1:]
    numBins = int(len(URIs)/blocks_per_bin)
    mass_array_bins =  np.loadtxt(folder + "/massArray_bins.csv", delimiter=',')
    
    localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
    restraints = pd.read_csv(localization_restraints_file,sep = ',')
    len_distribution = np.zeros((len(URIs)))
    n=0
    l0 = restraints['cell_length'].values[0]
    for i in range(numBins):
        l = restraints['cell_length'].values[i]
        len_distribution[i*blocks_per_bin:(i+1)*blocks_per_bin] = 2*l0/l**2
            
    
    distanceMat = np.zeros((N,N))
    distanceMatTmp = np.zeros((copyNum*N,copyNum*N))
    for i,U in enumerate(URIs):
        # print("%d/%d" % (i, len(URIs)))
        frame = load_URI(U)
        # print("load URI %d accumulated runtime: %f" % (i,time.time() - st))
        positions = frame["pos"]
        
        distanceMatTmp = distance_matrix(positions, positions)
        
        zeroMass_inds = np.where(mass_array_bins[:,int(np.floor(i/blocks_per_bin))]==0)
        distanceMatTmp[:,zeroMass_inds] = np.nan
        distanceMatTmp[zeroMass_inds,:] = np.nan
        
        for j in range(N):
            tmp = np.zeros((copyNum,copyNum*N))
            for k in range(copyNum):
                tmp[k,:] = distanceMatTmp[k*N+j,:]
            tmp = tmp.reshape((copyNum*copyNum,N))
            distanceMat[:,j] =  distanceMat[:,j] + np.nanmin(tmp[:,:N],axis=0)*len_distribution[i]
           
    distanceMat = distanceMat/np.sum(len_distribution)
    return distanceMat

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
        # list_models.remove(m)
        rm_list.append(i)
# list_models = list_models[:2]
rm_list.sort(reverse=True)
for i in rm_list:
    modelsList.pop(i)  

# modelsList = modelsList[:16]

URIs = list_URIs(modelsList[0])
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

blocks_per_bin = 10


args = []
for m in modelsList:
    args.append((m,blocks_per_bin,N,copyNum))


with Pool(cpu_count()) as p:
    Results = p.starmap(get_instance_distanceMat, args)

print('combining contacts\n')
combinedDistanceMat = np.zeros((N,N))
for r in Results:
    combinedDistanceMat += r
combinedDistanceMat = combinedDistanceMat/len(modelsList)

if '/' in modelsPath:
    tmp = modelsPath.split('/')
    modelsName = tmp[-1]
else:
    modelsName = modelsPath

np.save('Distance_maps/Data/simDistMat_%s.npy' % modelsName, combinedDistanceMat)



