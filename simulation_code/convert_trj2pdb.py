#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 15:03:32 2022

@author: Dvir Schirman

Convert polychrom's h5 block to a pdb trajectory to be read in pyMol.
Structures are rotated to minimize RDMS compared to the first frame.
"""

import polychrom
from polychrom.hdf5_format import list_URIs, load_URI
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align
import matplotlib.pyplot as plt
import pandas as pd
import os
from multiprocessing import Pool

N_MG1655 = 4641652

# pathToURIs = "241022_4Paper_normalModel/run_000/"
pathToURIs = "Ensemble_models/241022_4Paper_randomAll_20241022/run_000/"

# pathToURIs = "full_cell_cycle_model_2D_pooledExp_multiInstancesGPU_repInput_angK2_N1k_230908/run_000/"
# pathToURIs = "test/"
frameSkip=1
# outfilePrefix = pathToURIs[:-1]
# outfilePrefix = "full_cell_cycle_model_200k_2D_repInput_pooledExp"

# outfilePrefix = "241022_4Paper_normalModel"
outFolder = "pdbs"
if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
outfilePrefix = "241022_4Paper_randomAll"
outfilePrefix = '%s/%s' % (outFolder, outfilePrefix) 

all_URIs = list_URIs(pathToURIs)
all_URIs = all_URIs[1:]
first_frame = load_URI(all_URIs[0])
time = first_frame['time']
atm_name = "CA"

nMol = len(first_frame["pos"])
chains = [(0,nMol,True)]

first_frame_u = mda.Universe.empty(nMol,nMol,1,np.ones([nMol]))
first_frame_u.add_TopologyAttr('masses')
first_frame_u.atoms.masses = np.ones([nMol])
first_frame_u.load_new(first_frame["pos"])

restypes = np.array([[i]*(j[1]-j[0]) for i,j in enumerate(chains,start=1)]).flatten()
atom_num_list = np.arange(1,nMol+1)

mass_array_bins =  np.loadtxt(pathToURIs + "massArray_bins.csv", delimiter=',')
localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
restraints = pd.read_csv(localization_restraints_file,sep = ',')
num_of_bins = len(restraints)
blocks_per_bin = int(len(all_URIs)/num_of_bins)

with open(f"{outfilePrefix}.pdb",'w') as of:
        for i,frames in enumerate(all_URIs):
        # for i in range(1):
            # frames = all_URIs[-1]
            if np.mod(i,frameSkip)>0:
                continue
            if np.mod(i,100)==0:
                print(i)
            frame = load_URI(frames)
            time = frame["time"]
            
            u = mda.Universe.empty(nMol,nMol,1,np.ones([nMol]))
            u.add_TopologyAttr('masses')
            
            bin_num = int(np.floor(i/blocks_per_bin))
            mass_array = mass_array_bins[:,bin_num]
            u.atoms.masses = mass_array
                
            u.load_new(frame["pos"])

            u_center = u.atoms.center_of_mass()
            u_ref_c = first_frame_u.atoms.positions - first_frame_u.atoms.center_of_mass()
            u_c = u.atoms.positions - u_center
            #R, rmsd = align.rotation_matrix(u_c, u_ref_c)
            # print(rmsd)
            
            # u.atoms.translate(-u_center)
            # u.atoms.rotate(R)
            # u.atoms.translate(u_center)
            
            of.write(f"Coordinates for t={time:.3f}\n");
            of.write(f"{nMol}\n");
            # replicated_inds_old_chr = np.where(mass_array[4000:]>0)[0]
            replicated_inds_chr1 = np.where(mass_array[int(nMol/4):int(2*nMol/4)]>0)[0]+1
            
            for i,coords in enumerate(u.atoms.positions,start=1):
                if mass_array[i-1]:
                    if i<=nMol/4:
                        if i in replicated_inds_chr1:
                            of.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                                "ATOM", i, atm_name, " ", "ALA", "B", i, " ", coords[0],coords[1],coords[2], 1, 1, " ", " "))
                        else:
                            of.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                                "ATOM", i, atm_name, " ", "ALA", "A", i, " ", coords[0],coords[1],coords[2], 1, 1, " ", " "))
                    elif i>nMol/4 and i<=2*nMol/4:
                        of.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                            "ATOM", i, atm_name, " ", "ALA", "C", i, " ", coords[0],coords[1],coords[2], 1, 1, " ", " "))
                    elif i>2*nMol/4 and i<=3*nMol/4:
                        of.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                            "ATOM", i, atm_name, " ", "ALA", "D", i, " ", coords[0],coords[1],coords[2], 1, 1, " ", " "))
                    else:
                        of.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                            "ATOM", i, atm_name, " ", "ALA", "E", i, " ", coords[0],coords[1],coords[2], 1, 1, " ", " "))
                #else:
                    #of.write("{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n".format(
                        #"ATOM", i, atm_name, " ", "ALA", "Z", i, " ", coords[0],coords[1],coords[2], 1, 1, " ", " "))
                    
            of.write(f"ENDMDL\n") 



