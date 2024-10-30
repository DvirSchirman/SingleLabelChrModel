#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:38:08 2024

@author: dvirs
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import os
#from multiprocessing import Pool, cpu_count
import seaborn as sb
import glob
import argparse
import matplotlib as mpl
from polychrom.hdf5_format import list_URIs, load_URI

def create_parser():
    parser = argparse.ArgumentParser(description='Plot a bulk distance map from an ensemble of models')
    parser.add_argument("--CisTrans", action='store_true', help="Add flag for defrentiating between chromosomal interactions in-Cis and in-Trans")
    parser.add_argument('-m', '--model_path', default='241022_4Paper_normalModel',type=str, help="path to model output")
    parser.add_argument('-w', '--window_size', default=1,type=int, help="sliding window size")
    parser.add_argument("--plotRiboOp", action='store_true', help="overlay ribosomal operons sites that were shown to affect chromosome macrodomains boundaries (Lioy 2018)ontop of distance mat.")
    
    return parser
    
parser=create_parser()
args = parser.parse_args()

modelsPath = args.model_path
window_size = args.window_size

N_MG1655 = 4641652
N = 1000
copyNum = 4
blocks_per_bin = 10
oriC_site = 3925860
terBead = int((oriC_site-N_MG1655/2)/N_MG1655*N)
oriBead = int(oriC_site/N_MG1655*N)

beadsToGenome = list(map(float,np.linspace(0,N_MG1655/1e6,N)))
beadsToGenome = list(np.round(np.array(beadsToGenome),2))

MD = np.array([3759738, 4641652])
MD = MD/N_MG1655*N
oriMD = MD.astype('int')
MD = np.array([603414, 1206829])
MD = MD/N_MG1655*N
rightMD = MD.astype('int')
MD = np.array([2042326+1, 2877824])
MD = MD/N_MG1655*N
leftMD = MD.astype('int')
MD = np.array([1206829+1, 2042326])
MD = MD/N_MG1655*N
terMD = MD.astype('int')

LioyBoundaries = np.array([1.304e6, 1.7095e6, 2.730e6, 3.428e6])
LioyBoundaries = LioyBoundaries/N_MG1655*N

volume_ratio = 0.1
r_monomer = 0.5 #never change
v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
R=L/2 #radius of cylinder
L = round(L,1)
R = round(R,1)
scaling_factor = 1/L

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

URIs = list_URIs(modelsList[0])
numBins = int(len(URIs)/blocks_per_bin)
numModels = len(modelsList)

thr = []
thrL = []
thrT = []
thrTL = []

thrValH = 0.99
thrValL = 0.01
lineWidth = 10

if '/' in modelsPath:
        tmp = modelsPath.split('/')
        modelsPath = tmp[-1]

if args.CisTrans:
    combinedCisMat = np.load('Distance_maps/Data/simDistMatCisDynamic_%s.npy' % modelsPath)
    combinedTransMat = np.load('Distance_maps/Data/simDistMatTransDynamic_%s.npy' % modelsPath)
    
    combinedCisMat = combinedCisMat*scaling_factor
    combinedTransMat = combinedTransMat*scaling_factor
    
    outFolder = "Distance_maps/Figures/dynamicDistanceMat_CisTrans_%s" % modelsPath
    if args.plotRiboOp:
        outFolder = outFolder + 'riboOp'
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
        os.mkdir(outFolder + '/3DContacts')
    
    print('plotting contacts\n')
    
    cmap = mpl.cm.get_cmap("vlag_r").copy()
    cmap.set_bad('0')
    
    for i in range(int(np.floor(window_size/2)), numBins-int(np.floor(window_size/2))):
        thr.append(np.quantile(combinedCisMat[i],thrValH))
        thrL.append(np.quantile(combinedCisMat[i],thrValL))
        thrT.append(np.quantile(combinedTransMat[i],thrValH))
        thrTL.append(np.quantile(combinedTransMat[i],thrValL))
        
    thr = np.mean(np.array(thr))
    thrL = np.mean(np.array(thrL))
    thrT = np.mean(np.array(thrT))
    thrTL = np.mean(np.array(thrTL))
    
    for i in range(int(np.floor(window_size/2)), numBins-int(np.floor(window_size/2))):
        CisMatdf = pd.DataFrame(combinedCisMat[i],columns=beadsToGenome,index=beadsToGenome)
        fig, (ax1) = plt.subplots(1, 1)
        sb.heatmap(CisMatdf,vmin=thrL,vmax=thr,cmap=cmap,cbar_kws={'label': r'Pairwise distance [$\mu$m]'}, square=True, ax=ax1)
        ax1.axvline(x=terBead,linewidth=1,color='tomato')
        ax1.axvline(x=oriBead, linewidth=1, color='goldenrod')
        ax1.axhline(y=-2*lineWidth, xmin=(oriMD[0]+1.75*lineWidth)/N, xmax=(oriMD[1]-1.75*lineWidth)/N, color='goldenrod',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(terMD[0]+1.75*lineWidth)/N, xmax=(terMD[1]-1.75*lineWidth)/N, color='tomato',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(rightMD[0]+1.75*lineWidth)/N, xmax=(rightMD[1]-1.75*lineWidth)/N, color='plum',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(leftMD[0]+1.75*lineWidth)/N, xmax=(leftMD[1]-1.75*lineWidth)/N, color='mediumslateblue',linewidth=lineWidth, alpha=0.75)
        if args.plotRiboOp:
            for b in LioyBoundaries:
                ax1.axvline(x=b,linewidth=1,linestyle='--',color='black')
        ax1.set_ylim(1000, -4*lineWidth)
        tmp = ax1.get_yticks()
        tmp = tmp[::2]
        ax1.set_yticks(tmp)
        ax1.set_xticks(ax1.get_yticks())
        ax1.set_xticklabels(ax1.get_yticklabels())
        ax1.set_xlabel('Genome position [Mbp]')
        ax1.set_ylabel('Genome position [Mbp]')
        plt.savefig('%s/3DContacts/simDynamicCisMat%d.png' % (outFolder, i), bbox_inches='tight')
        if i==24:
            plt.savefig('%s/3DContacts/simDynamicCisMat%d.pdf' % (outFolder, i), bbox_inches='tight')
        # plt.show()
        plt.close(fig)
        
        TransMatdf = pd.DataFrame(combinedTransMat[i],columns=beadsToGenome,index=beadsToGenome)
        fig, (ax1) = plt.subplots(1, 1)
        sb.heatmap(TransMatdf,vmin=thrL,vmax=thr,cmap=cmap,cbar_kws={'label': r'Pairwise distance [$\mu$m]'}, square=True, ax=ax1)
        ax1.axvline(x=terBead,linewidth=1,color='tomato')
        ax1.axvline(x=oriBead, linewidth=1, color='goldenrod')
        ax1.axhline(y=-2*lineWidth, xmin=(oriMD[0]+1.75*lineWidth)/N, xmax=(oriMD[1]-1.75*lineWidth)/N, color='goldenrod',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(terMD[0]+1.75*lineWidth)/N, xmax=(terMD[1]-1.75*lineWidth)/N, color='tomato',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(rightMD[0]+1.75*lineWidth)/N, xmax=(rightMD[1]-1.75*lineWidth)/N, color='plum',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(leftMD[0]+1.75*lineWidth)/N, xmax=(leftMD[1]-1.75*lineWidth)/N, color='mediumslateblue',linewidth=lineWidth, alpha=0.75)
        if args.plotRiboOp:
            for b in LioyBoundaries:
                ax1.axvline(x=b,linewidth=1,linestyle='--',color='black')
        ax1.set_ylim(1000, -4*lineWidth)
        tmp = ax1.get_yticks()
        tmp = tmp[::2]
        ax1.set_yticks(tmp)
        ax1.set_xticks(ax1.get_yticks())
        ax1.set_xticklabels(ax1.get_yticklabels())
        ax1.set_xlabel('Genome position [Mbp]')
        ax1.set_ylabel('Genome position [Mbp]')
        plt.savefig('%s/3DContacts/simDynamicTransMat%d.png' % (outFolder, i), bbox_inches='tight')
        # plt.show()
        plt.close(fig)

else:
    combinedDistanceMat = np.load('simDistMatDynamic_%s.npy'% modelsPath)
    combinedDistanceMat = combinedDistanceMat*scaling_factor
    
    print('plotting contacts\n')
    
    outFolder = "Distance_maps/Figures/dynamicDistanceMat_%s" % modelsPath
    if args.plotRiboOp:
        outFolder = outFolder + 'riboOp'
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
        os.mkdir(outFolder + '/3DContacts')
    
    for i in range(int(np.floor(window_size/2)), numBins-int(np.floor(window_size/2))):
        thr.append(np.quantile(combinedDistanceMat[i],thrValH))
        thrL.append(np.quantile(combinedDistanceMat[i],thrValL))
    thr = np.mean(np.array(thr))
    thrL = np.mean(np.array(thrL))
    for i in range(int(np.floor(window_size/2)), numBins-int(np.floor(window_size/2))):
        distanceMatdf = pd.DataFrame(combinedDistanceMat[i],columns=beadsToGenome,index=beadsToGenome)
        fig, (ax1) = plt.subplots(1, 1)
        sb.heatmap(distanceMatdf,vmin=thrL,vmax=thr,cmap='vlag_r',cbar_kws={'label': r'Pairwise distance [$\mu$m]'}, square=True, ax=ax1)
        ax1.axvline(x=terBead,linewidth=1,color='tomato')
        ax1.axvline(x=oriBead, linewidth=1, color='goldenrod')
        ax1.axhline(y=-2*lineWidth, xmin=(oriMD[0]+1.75*lineWidth)/N, xmax=(oriMD[1]-1.75*lineWidth)/N, color='goldenrod',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(terMD[0]+1.75*lineWidth)/N, xmax=(terMD[1]-1.75*lineWidth)/N, color='tomato',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(rightMD[0]+1.75*lineWidth)/N, xmax=(rightMD[1]-1.75*lineWidth)/N, color='plum',linewidth=lineWidth, alpha=0.75)
        ax1.axhline(y=-2*lineWidth, xmin=(leftMD[0]+1.75*lineWidth)/N, xmax=(leftMD[1]-1.75*lineWidth)/N, color='mediumslateblue',linewidth=lineWidth, alpha=0.75)
        if args.plotRiboOp:
            for b in LioyBoundaries:
                ax1.axvline(x=b,linewidth=1,linestyle='--',color='black')
        ax1.set_ylim(1000, -4*lineWidth)
        tmp = ax1.get_yticks()
        tmp = tmp[::2]
        ax1.set_yticks(tmp)
        ax1.set_xticks(ax1.get_yticks())
        ax1.set_xticklabels(ax1.get_yticklabels())
        ax1.set_xlabel('Genome position [Mbp]')
        ax1.set_ylabel('Genome position [Mbp]')
        plt.savefig('%s/3DContacts/simDynamicDistMat%d.png' % (outFolder, i), bbox_inches='tight')
        # plt.show()
        plt.close(fig)
