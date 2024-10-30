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

def create_parser():
    parser = argparse.ArgumentParser(description='Plot a bulk distance map from an ensemble of models')
    parser.add_argument("--CisTrans", action='store_true', help="Add flag for defrentiating between chromosomal interactions in-Cis and in-Trans")
    parser.add_argument('-m', '--model_path', default='Ensemble_models/241022_4Paper_normalModel',type=str, help="path to model output")
    return parser
    
parser=create_parser()
args = parser.parse_args()

modelsPath = args.model_path

N_MG1655 = 4641652
N = 1000
copyNum = 4
oriC_site = 3925860
terBead = int((oriC_site-N_MG1655/2)/N_MG1655*N)
oriBead = int(oriC_site/N_MG1655*N)

volume_ratio = 0.1
r_monomer = 0.5 #never change
v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
R=L/2 #radius of cylinder
L = round(L,1)
R = round(R,1)
scaling_factor = 1/L

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

lineWidth = 10

if '/' in modelsPath:
        tmp = modelsPath.split('/')
        modelsPath = tmp[-1]

if args.CisTrans:
    combinedCisMat = np.load('Distance_maps/Data/simDistMatCis_%s.npy' % modelsPath)
    combinedTransMat = np.load('Distance_maps/Data/simDistMatTrans_%s.npy' % modelsPath)
    
    print('plotting contacts\n')
    
    combinedCisMat = combinedCisMat*scaling_factor
    combinedTransMat = combinedTransMat*scaling_factor
    
    cmap = mpl.cm.get_cmap("vlag_r").copy()
    cmap.set_bad('0')

    cisMatdf = pd.DataFrame(combinedCisMat,columns=beadsToGenome,index=beadsToGenome)
    # plt.figure
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(cisMatdf,vmin=np.quantile(cisMatdf,0.01),vmax=np.quantile(cisMatdf,0.99),cmap=cmap,cbar_kws={'label': r'Pairwise distance [$\mu$m]'}, square=True, ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='tomato')
    ax1.axvline(x=oriBead, linewidth=1, color='goldenrod')
    ax1.axhline(y=-2*lineWidth, xmin=(oriMD[0]+1.75*lineWidth)/N, xmax=(oriMD[1]-1.75*lineWidth)/N, color='goldenrod',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(terMD[0]+1.75*lineWidth)/N, xmax=(terMD[1]-1.75*lineWidth)/N, color='tomato',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(rightMD[0]+1.75*lineWidth)/N, xmax=(rightMD[1]-1.75*lineWidth)/N, color='plum',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(leftMD[0]+1.75*lineWidth)/N, xmax=(leftMD[1]-1.75*lineWidth)/N, color='mediumslateblue',linewidth=lineWidth, alpha=0.75)
    ax1.set_ylim(1000, -4*lineWidth)
    tmp = ax1.get_yticks()
    tmp = tmp[::2]
    ax1.set_yticks(tmp)
    ax1.set_xticks(ax1.get_yticks())
    ax1.set_xticklabels(ax1.get_yticklabels())
    ax1.set_xlabel('Genome position [Mbp]')
    ax1.set_ylabel('Genome position [Mbp]')
    plt.savefig('Distance_maps/Figures/simDistMatCis_%s.png' % modelsPath, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    transMatdf = pd.DataFrame(combinedTransMat,columns=beadsToGenome,index=beadsToGenome)
    # plt.figure
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(transMatdf,vmin=np.quantile(transMatdf,0.01),vmax=np.quantile(transMatdf,0.99),cmap=cmap,cbar_kws={'label': r'Pairwise distance [$\mu$m]'}, square=True, ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='tomato')
    ax1.axvline(x=oriBead, linewidth=1, color='goldenrod')
    ax1.axhline(y=-2*lineWidth, xmin=(oriMD[0]+1.75*lineWidth)/N, xmax=(oriMD[1]-1.75*lineWidth)/N, color='goldenrod',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(terMD[0]+1.75*lineWidth)/N, xmax=(terMD[1]-1.75*lineWidth)/N, color='tomato',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(rightMD[0]+1.75*lineWidth)/N, xmax=(rightMD[1]-1.75*lineWidth)/N, color='plum',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(leftMD[0]+1.75*lineWidth)/N, xmax=(leftMD[1]-1.75*lineWidth)/N, color='mediumslateblue',linewidth=lineWidth, alpha=0.75)
    ax1.set_ylim(1000, -4*lineWidth)
    tmp = ax1.get_yticks()
    tmp = tmp[::2]
    ax1.set_yticks(tmp)
    ax1.set_xticks(ax1.get_yticks())
    ax1.set_xticklabels(ax1.get_yticklabels())
    ax1.set_xlabel('Genome position [Mbp]')
    ax1.set_ylabel('Genome position [Mbp]')
    plt.savefig('Distance_maps/Figures/simDistMatTrans_%s.png' % modelsPath, bbox_inches='tight')
    # plt.show()
    plt.close(fig)

else:
    combinedDistanceMat = np.load('Distance_maps/Data/simDistMat_%s.npy' % modelsPath)
    print('plotting contacts\n')
    
    combinedDistanceMat = combinedDistanceMat*scaling_factor
    distanceMatdf = pd.DataFrame(combinedDistanceMat,columns=beadsToGenome,index=beadsToGenome)
    # plt.figure
    fig, (ax1) = plt.subplots(1, 1)
    sb.heatmap(distanceMatdf,vmin=np.quantile(distanceMatdf,0.01),vmax=np.quantile(distanceMatdf,0.99),cmap='vlag_r',cbar_kws={'label': r'Pairwise distance [$\mu$m]'}, square=True, ax=ax1)
    ax1.axvline(x=terBead,linewidth=1,color='tomato')
    ax1.axvline(x=oriBead, linewidth=1, color='goldenrod')
    ax1.axhline(y=-2*lineWidth, xmin=(oriMD[0]+1.75*lineWidth)/N, xmax=(oriMD[1]-1.75*lineWidth)/N, color='goldenrod',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(terMD[0]+1.75*lineWidth)/N, xmax=(terMD[1]-1.75*lineWidth)/N, color='tomato',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(rightMD[0]+1.75*lineWidth)/N, xmax=(rightMD[1]-1.75*lineWidth)/N, color='plum',linewidth=lineWidth, alpha=0.75)
    ax1.axhline(y=-2*lineWidth, xmin=(leftMD[0]+1.75*lineWidth)/N, xmax=(leftMD[1]-1.75*lineWidth)/N, color='mediumslateblue',linewidth=lineWidth, alpha=0.75)
    ax1.set_ylim(1000, -4*lineWidth)
    tmp = ax1.get_yticks()
    tmp = tmp[::2]
    ax1.set_yticks(tmp)
    ax1.set_xticks(ax1.get_yticks())
    ax1.set_xticklabels(ax1.get_yticklabels())
    ax1.set_xlabel('Genome position [Mbp]')
    ax1.set_ylabel('Genome position [Mbp]')
    plt.savefig('Distance_maps/Figures/simDistMat_%s.png' % modelsPath, bbox_inches='tight')
    # plt.savefig('simDistMat_%s.pdf' % modelsPath, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
