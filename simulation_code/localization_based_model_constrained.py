#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 16:19:15 2022

@author: Dvir Schirman

This simulation uses polychrom to create a polymer model of a whole bacterial chromosome
as it segregates during the cell cycle while until cell division.
The input to the model is localization data for specific loci on the chromosome obtained from fluoresence microscopy data
from E. coli strains with chromosome labels (either parS or malIx12) integrated at specific sites on the genome.
The data can be obtained in single strain experiments or pooled library with demultiplexing based on in-situ genotyping.
"""

import os, sys
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
import openmm
import os
from polychrom.hdf5_format import HDF5Reporter, list_URIs
# import polymerutils
import numpy as np
import polychrom.hdf5_format
import pandas as pd
from openmm.app import PDBFile, PDBReporter
import time
from multiprocessing import Pool, cpu_count
import itertools
import glob
import argparse
import random
from matplotlib import pyplot as plt

def create_parser():
    parser = argparse.ArgumentParser(description='Run simulations of a replicating chromosome with localization based restraints.')
    parser.add_argument("--no_restraints", action='store_true', help="Add flag if want no localilzation based restraints, ie just replication")
    parser.add_argument("--use_cpu", action='store_true', help="Use CPU instead of GPU, good for debug on a platform with no GPU")
    parser.add_argument("--no_radial", action='store_true', help="Add flag if want no radial coordinates restraints.")
    parser.add_argument("--randomize_radial", action='store_true', help="Add flag to randomize radial coordinates restraints.")
    parser.add_argument("--randomize_all", action='store_true', help="Add flag to randomize all coordinates restraints.")
    parser.add_argument('-x', '--leave_out', default=0,type=int, help="Number of loci to leave out of the model (randomly chosen)")
    parser.add_argument('-d', '--leave_out_all_ind', default=-1,type=int, help="Leave out index. For Leaving out all loci in 10 different sets")
    parser.add_argument('-c', '--col_rate', default=0.03,type=float, help="Collision rate that sets drag force")
    parser.add_argument('-t', '--trunc', default=1.5,type=float, help="trunc parameter for strength of excluded volume")
    parser.add_argument('-a', '--angular_k', default=2.0,type=float, help="Angular hormonic bond spring coeficient")
    parser.add_argument('-N', '--num_beads', default=1000,type=int, help="Number of simulated monomers per chormosome copy")
    parser.add_argument('-n', '--num_instances', default=100,type=int, help="The number of simulation instances")
    parser.add_argument('-w', '--num_nodes', default=1,type=int, help="The number of cluster nodes running the simulation")
    parser.add_argument('-i', '--node_ind',type=int, help="Node index (if number of nodes is >1 must supply index)")
    parser.add_argument('-l', '--time_per_step', default=0.0018,type=float, help="real time (in seconds) per simulation step")
    parser.add_argument('-r', '--restraints_file', default='../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv',type=str, help="Input restraints file")
    parser.add_argument('-o', '--output_folder_prefix', default='simOut',type=str, help="Input restraints file")
    parser.add_argument('-e', '--tether_exp', default=1.0,type=float, help="Exponent parameter for the localization tethering restraints")
    parser.add_argument('-k', '--tether_alpha', default=1e-4,type=float, help="Scaling parameter for the localization tethering restraints")
    parser.add_argument('-b', '--blocks_per_bin', default=10,type=int, help="Number of reported blocks per cell size bin")
    parser.add_argument('-v', '--volume_ratio', default=0.1,type=float, help="The ratio of total monomers volume of one chromosome copy to the total volume of the cell")
    parser.add_argument('-s', '--seed', default=1,type=int, help="seed for random number generator, use the same seed for multiple instances of the same randomized model")
    return parser

def create_phantom_polymer(polymer, boxSize):
    phantom_polymer = np.zeros(polymer.shape)
    for i in range(len(phantom_polymer)):
        # phantom_polymer[i,0] = polymer[i,0]+boxSize[0]
        phantom_polymer[i,0] = polymer[i,0] + 1
        phantom_polymer[i,1:] = polymer[i,1:]
    # phantom_polymer = polymer
    return np.concatenate((polymer,phantom_polymer))
  
def create_phantom_polymer2(polymer, boxSize):
    phantom_polymer = np.zeros(polymer.shape)
    for i in range(len(phantom_polymer)):
        # phantom_polymer[i,1] = polymer[i,1]+boxSize[1]
        phantom_polymer[i,1] = polymer[i,1]+1
        phantom_polymer[i,0] = polymer[i,0]
        phantom_polymer[i,2] = polymer[i,2]
    return np.concatenate((polymer,phantom_polymer))

def clean_polychrom_folder(folder):
    file_list = glob.glob(folder + '/*')
    # print(len(file_list))
    file_list = [i for i in file_list if 'blocks' not in i]
    file_list = [i for i in file_list if 'massArray' not in i]
    # print(len(file_list))
    for f in file_list:
        os.remove(f)

def run_sim(n, folder, args):
  
  if args.use_cpu:
      platform = "CPU"
  else:
      platform = "OPENCL"
  
  
  N_MG1655 = 4641652
  oriC_site = 3925860       
  N=args.num_beads
  #time_per_step = 0.00005 # Seconds
  time_per_step = args.time_per_step # Seconds
  #time_per_step = 0.001 # Seconds
  generation_time = 60*60 # Seconds
  # ~ steps_per_bin = 200000
  blocks_per_bin = args.blocks_per_bin
  # steps_per_bin = 100
  # blocks_per_bin = 1
  angK = args.angular_k
  tetherExp = args.tether_exp
  tetherAlpha = args.tether_alpha
  tetherAlphaR = 50*tetherAlpha
  tetherAlphaRnoInput = 0.025
  trunc_val = np.sqrt(args.trunc)
  
  if args.no_radial:
      tetherAlphaR = 0
      tetherAlphaRnoInput = 0
  
  instance_folder = '%s/run_%03d' % (folder, n)
  kMax = 5000

  volume_ratio = args.volume_ratio
  r_monomer = 0.5 #never change
  v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
  v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
  L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
  R=L/2 #radius of cylinder
  L = round(L,1)
  R = round(R,1)

  localization_restraints_file = args.restraints_file
  restraints = pd.read_csv(localization_restraints_file,sep = ',')
  
  np.random.seed(args.seed)
  if args.randomize_all:
      args.randomize_radial = True
      Ltbl = restraints.filter(regex='mu\d')
      Lvalues = Ltbl.values
      
      # Lvalues = Lvalues.reshape(Ltbl.shape[0]*Ltbl.shape[1])
      # Lvalues = Lvalues[~np.isnan(Lvalues)]
      # Lvalues = np.random.permutation(Lvalues)
      # ~ n=0
      # ~ for i in range(Ltbl.shape[0]):
          # ~ for j in range(Ltbl.shape[1]):
              # ~ if ~np.isnan(Ltbl.iloc[i,j]):
                  # ~ Ltbl.iloc[i,j] = Lvalues[n]
                  # ~ n+=1
      Lmin = np.nanmin(Lvalues)
      Lmax = np.nanmax(Lvalues)
      for i in range(Ltbl.shape[0]):
          for j in range(Ltbl.shape[1]):
              if Ltbl.iloc[i,j]<-0.1:
                  Ltbl.iloc[i,j] = np.random.uniform(Lmin-0.05, -0.05)
              elif Ltbl.iloc[i,j]>=-0.1 and Ltbl.iloc[i,j]<=0.1:
                  Ltbl.iloc[i,j] = np.random.uniform(-0.15, 0.15)
              elif Ltbl.iloc[i,j]>0.1:
                  Ltbl.iloc[i,j] = np.random.uniform(0.05, Lmax+0.05)
                  
      restraints.update(Ltbl)
  
  if args.randomize_radial:
      Rtbl = restraints.filter(regex='muR')
      Rvalues = Rtbl.values
      Rvalues = Rvalues.reshape(Rtbl.shape[0]*Rtbl.shape[1])
      Rvalues = np.random.permutation(Rvalues)
      Rvalues = Rvalues.reshape(Rtbl.shape[0],Rtbl.shape[1])
      for i in range(Rtbl.shape[1]):
          Rtbl.loc[:,Rtbl.columns[i]] = Rvalues[:,i]
      restraints.update(Rtbl)
  
  #if args.leave_out > 0:
      
  # dropInd = 1
  # restraints.drop(restraints.columns[10*dropInd+1:10*(dropInd+1)+1],axis=1,inplace=True)
  num_of_bins = len(restraints)
  steps_per_bin = int(generation_time/num_of_bins/time_per_step)
  genome_loci = restraints.columns[1:]
  genome_loci = [int(a.split(sep='_')[0]) for a in genome_loci]
  genome_loci = np.unique(genome_loci)
  
  if args.leave_out > 0:
      THR = N_MG1655/25
      while True:
          drop_genome_loci = np.random.choice(genome_loci,args.leave_out,replace=False)
          drop_genome_loci.sort()
          if np.min(np.diff(drop_genome_loci))<THR or (N_MG1655-drop_genome_loci[-1]+drop_genome_loci[0])<THR:
              continue
          else:
              break
      for g in drop_genome_loci:
          restraints = restraints.drop(list(restraints.filter(regex=str(g))),axis=1)
          genome_loci = genome_loci[genome_loci!=g]
      if not os.path.isdir(folder):
          os.mkdir(folder)
      np.savetxt(folder + '/dropped_loci.txt',drop_genome_loci,fmt='%d')
      
  if args.leave_out_all_ind >= 0:
      drop_genome_loci = genome_loci[range(args.leave_out_all_ind, len(genome_loci),10)]
      for g in drop_genome_loci:
          restraints = restraints.drop(list(restraints.filter(regex=str(g))),axis=1)
          genome_loci = genome_loci[genome_loci!=g]
      np.savetxt(folder + '/dropped_loci.txt',drop_genome_loci,fmt='%d')
          
  grid = np.linspace(1,N_MG1655,N,dtype=int)
  bead_num = np.array([np.argmin(abs(grid-i)) for i in genome_loci])
  oriC_bead_num = np.argmin(abs(grid-oriC_site))

  bin0_restraints = pd.DataFrame(genome_loci, columns=['genome_location'])
  bin0_restraints['Z_mu'] = [restraints[col][0] for col in restraints.columns if 'mu1' in col]
  bin0_restraints['Z_sigma'] = [restraints[col][0] for col in restraints.columns if 'sigma1' in col]
  bin0_restraints['bead_num'] = bead_num

  reporter = HDF5Reporter(folder=instance_folder, max_data_length=10000, overwrite=True)
  mass_array = 100*np.concatenate((np.ones(N), np.zeros(N)))
  mass_array = np.concatenate((mass_array, np.zeros(N*2)))

  pole_site = [oriC_site - round(N_MG1655/4,0), np.round(N_MG1655/4,0) - (N_MG1655 - oriC_site)]
  cell_side = []
  for g in genome_loci:
    if g>=pole_site[0] or g<pole_site[1]:
      cell_side.append('F')
    else:
      cell_side.append('B')
  bin0_restraints['side'] = cell_side

  print(bin0_restraints)

  polymer = None
  # starting conformation algorithm might result in a an error due to "bad" random assignment, try again until it succeed 
  print("start polymer starting conformation")
  while polymer is None:
    try:
      polymer = starting_conformations.restrained_rect_box_1D(N, [1.6*R,1.6*R,0.8*L], bin0_restraints)
      # polymer = starting_conformations.grow_rect_box(400, [9,9,40], method='extended')
    except:
      pass
  print("completed polymer starting conformation")

  polymer = create_phantom_polymer(polymer,[1.6*R,1.6*R,0.8*L])
  polymer = create_phantom_polymer2(polymer,[1.6*R,1.6*R,0.8*L])
  N = N*4
  mass_array_bins = np.zeros(shape=(N,num_of_bins))
  
  tmp_dist = np.zeros(shape=(len(bead_num),2))
  tmp_dist[:,0] = np.abs(oriC_bead_num-bead_num)
  tmp_dist[:,1] = np.mod(bead_num - oriC_bead_num,int(N/4))
  sites_ori_dist = np.min(tmp_dist,axis=1)
  maxOriDist = max(sites_ori_dist)
  #%%
  dbg_bin=21
  for i in range(num_of_bins):
      mu2 = np.array([restraints[col][i] for col in restraints.columns if 'mu2' in col])
      segregated_sites = bead_num[np.where(np.isfinite(mu2))]
      tmp_dist = np.zeros(shape=(len(segregated_sites),2))
      tmp_dist[:,0] = np.abs(oriC_bead_num-segregated_sites)
      tmp_dist[:,1] = np.mod(segregated_sites - oriC_bead_num,int(N/4))
      segregated_sites_ori_dist = np.min(tmp_dist,axis=1)
      # segregated_sites_ori_dist = [np.abs(oriC_bead_num-s) if s>oriC_bead_num/2 else np.mod(s - oriC_bead_num,int(N/2)) for s in segregated_sites]
      if segregated_sites_ori_dist.any():
          segregated_sites_num = max(segregated_sites_ori_dist)*2
      else:
          segregated_sites_num = 0
      if max(segregated_sites_ori_dist) >= maxOriDist:
          segregated_sites_num = int(N/4)
      segregated_array = np.mod(np.arange(oriC_bead_num-segregated_sites_num/2,oriC_bead_num+segregated_sites_num/2),int(N/4))+int(N/4)
      segregated_array = segregated_array.astype(int)
      #if i==dbg_bin:
        #print(segregated_array)
        #print(segregated_sites_ori_dist)
        #wait = input("Press Enter to continue.")
      mass_array[segregated_array] = 100
      mass_array_bins[:,i] = mass_array
      segregated_sites_2 = segregated_sites
      segregated_array_2 = segregated_array
      
      mu3 = np.array([restraints[col][i] for col in restraints.columns if 'mu3' in col])
      segregated_sites = bead_num[np.where(np.isfinite(mu3))]
      tmp_dist = np.zeros(shape=(len(segregated_sites),2))
      tmp_dist[:,0] = np.abs(oriC_bead_num-segregated_sites)
      tmp_dist[:,1] = np.mod(segregated_sites - oriC_bead_num,int(N/4))
      segregated_sites_ori_dist = np.min(tmp_dist,axis=1)
      # segregated_sites_ori_dist = [np.abs(oriC_bead_num-s) if s>oriC_bead_num/2 else np.mod(s - oriC_bead_num,int(N/2)) for s in segregated_sites]
      if segregated_sites_ori_dist.any():
        segregated_sites_num = max(segregated_sites_ori_dist)*2
      else:
        segregated_sites_num = 0
      segregated_array = np.mod(np.arange(oriC_bead_num-segregated_sites_num/2,oriC_bead_num+segregated_sites_num/2),int(N/4))+2*int(N/4)
      segregated_array = segregated_array.astype(int)
      #if i==dbg_bin:
        #print(segregated_array)
        #print(segregated_sites_ori_dist)
        #wait = input("Press Enter to continue.")
      mass_array[segregated_array] = 100
      mass_array_bins[:,i] = mass_array
      segregated_sites_3 = segregated_sites
      segregated_array_3 = segregated_array
      
      mu4 = np.array([restraints[col][i] for col in restraints.columns if 'mu4' in col])
      segregated_sites = bead_num[np.where(np.isfinite(mu4))]
      tmp_dist = np.zeros(shape=(len(segregated_sites),2))
      tmp_dist[:,0] = np.abs(oriC_bead_num-segregated_sites)
      tmp_dist[:,1] = np.mod(segregated_sites - oriC_bead_num,int(N/4))
      segregated_sites_ori_dist = np.min(tmp_dist,axis=1)
      # segregated_sites_ori_dist = [np.abs(oriC_bead_num-s) if s>oriC_bead_num/2 else np.mod(s - oriC_bead_num,int(N/2)) for s in segregated_sites]
      if segregated_sites_ori_dist.any():
        segregated_sites_num = max(segregated_sites_ori_dist)*2
      else:
        segregated_sites_num = 0
      segregated_array = np.mod(np.arange(oriC_bead_num-segregated_sites_num/2,oriC_bead_num+segregated_sites_num/2),int(N/4))+3*int(N/4)
      segregated_array = segregated_array.astype(int)
      #if i==dbg_bin:
        #print(segregated_array)
        #print(segregated_sites_ori_dist)
        #wait = input("Press Enter to continue.")
      mass_array[segregated_array] = 100
      mass_array_bins[:,i] = mass_array
      segregated_sites_4 = segregated_sites
      segregated_array_4 = segregated_array
      
      trunc_array = trunc_val*mass_array/100
      
      if i>0:
          current_coordinates = sim.get_data()
          newly_replicated = np.where(mass_array_bins[:,i]!=mass_array_bins[:,i-1])[0]  
          newly_replicated2 = newly_replicated[np.logical_and(newly_replicated>=N/4, newly_replicated<2*N/4)]
          newly_replicated3 = newly_replicated[np.logical_and(newly_replicated>=2*N/4, newly_replicated<3*N/4)]
          newly_replicated4 = newly_replicated[np.logical_and(newly_replicated>=3*N/4, newly_replicated<N)]
          
          current_coordinates[newly_replicated2,0] = current_coordinates[newly_replicated2-int(N/4),0] - np.sign(current_coordinates[newly_replicated2-int(N/4),0])
          current_coordinates[newly_replicated2,1:2] = current_coordinates[newly_replicated2-int(N/4),1:2]
          current_coordinates[newly_replicated3,0] = current_coordinates[newly_replicated3-int(2*N/4),0] - np.sign(current_coordinates[newly_replicated3-int(2*N/4),0])
          current_coordinates[newly_replicated3,1:2] = current_coordinates[newly_replicated3-int(2*N/4),1:2]
          current_coordinates[newly_replicated4,0] = current_coordinates[newly_replicated4-int(2*N/4),0] - np.sign(current_coordinates[newly_replicated4-int(2*N/4),0])
          current_coordinates[newly_replicated4,1:2] = current_coordinates[newly_replicated4-int(2*N/4),1:2]
          
          unreplicated = np.where(mass_array_bins[:,i]==0)[0]
          unreplicated2 = unreplicated[np.logical_and(unreplicated>=N/4, unreplicated<2*N/4)]
          unreplicated3 = unreplicated[np.logical_and(unreplicated>=2*N/4, unreplicated<3*N/4)]
          unreplicated4 = unreplicated[np.logical_and(unreplicated>=3*N/4, unreplicated<N)]
          current_coordinates[unreplicated2,:] = current_coordinates[unreplicated2-int(N/4),:]
          current_coordinates[unreplicated3,:] = current_coordinates[unreplicated3-int(2*N/4),:]
          current_coordinates[unreplicated4,:] = current_coordinates[unreplicated4-int(2*N/4),:]
          
      sim = simulation.Simulation(
          platform=platform, 
          integrator="variableLangevin",
          error_tol=0.003,
          GPU="0",
          collision_rate=0.03,
          N=N,
          save_decimals=2,
          PBCbox=False,
          reporters=[reporter],
          mass = mass_array
          # length_scale = 1000,
      )
      
      if i==0:
          sim.set_data(polymer, center=True)
      else:
          sim.set_data(current_coordinates)
          
      added_length = (restraints.cell_length[i]-restraints.cell_length[0])/restraints.cell_length[0]*L
      sim.add_force(forces.cylindrical_confinement(sim, R, bottom=-L/2-added_length/2, top=L/2+added_length/2, k=1))
      #sim.add_force(forces.smooth_square_well())
      
      if not args.no_restraints:
          particles = np.unique(np.concatenate((bead_num,segregated_sites_2+int(N/4),segregated_sites_3+2*int(N/4), segregated_sites_4+3*int(N/4))))
          allParticles = np.where(mass_array_bins[:,i]>0)[0]
          position_vec = np.zeros((2,N))
          k_vec_all = np.zeros((2,N))
          for p in allParticles: 
              position = np.zeros(2)
              k_vec = np.zeros(2)
              if p<N/4:
                  if p in particles:
                      genome_locus = genome_loci[bead_num==p][0]
                      position[1] = restraints[str(genome_locus)+'_mu1'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                      k_vec[1] = tetherAlpha/(restraints[str(genome_locus)+'_sigma1'][i])**tetherExp
                      if k_vec[1] > kMax:
                          k_vec[1] = kMax
                      position[0] = restraints[str(genome_locus)+'_muR'][i]*R
                      k_vec[0] = tetherAlphaR/(restraints[str(genome_locus)+'_sigmaR'][i])**tetherExp
                      if k_vec[0] > kMax:
                          k_vec[0] = kMax
                  else:
                      position[0] = 0.2*R
                      k_vec[0] = tetherAlphaRnoInput
                      tmpP = particles[particles<N/4]
                      nearP1 = tmpP[tmpP<p]
                      if nearP1.any():
                          nearP1 = nearP1[-1]
                          nearG1 = genome_loci[bead_num==nearP1][0]
                          pos1 = restraints[str(nearG1)+'_mu1'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(nearG1)+'_sigma1'][i]
                      else:
                          nearP1 = tmpP[-1]
                          nearG1 = genome_loci[bead_num==nearP1][0]-N_MG1655
                          pos1 = restraints[str(genome_loci[bead_num==nearP1][0])+'_mu1'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(genome_loci[bead_num==nearP1][0])+'_sigma1'][i]
                          
                      nearP2 = tmpP[tmpP>p]
                      if nearP2.any():
                          nearP2 = nearP2[0]
                          nearG2 = genome_loci[bead_num==nearP2][0]
                          pos2 = restraints[str(nearG2)+'_mu1'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(nearG2)+'_sigma1'][i]
                      else:
                          nearP2 = tmpP[0]
                          nearG2 = N_MG1655+genome_loci[bead_num==nearP2][0]
                          pos2 = restraints[str(genome_loci[bead_num==nearP2][0])+'_mu1'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(genome_loci[bead_num==nearP2][0])+'_sigma1'][i]
                      
                      g=grid[p]
                      m = (pos2-pos1)/(nearG2-nearG1)
                      b = pos1-m*nearG1
                      pos = m*g + b
                      s = (s1+s2)/2
                      position[1] = pos
                      k_vec[1] = tetherAlpha/s**tetherExp
              
              elif p>=N/4 and p<2*N/4:
                  if p in particles:
                      genome_locus = genome_loci[bead_num==p-int(N/4)][0]
                      position[1] = restraints[str(genome_locus)+'_mu2'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                      k_vec[1] = tetherAlpha/(restraints[str(genome_locus)+'_sigma2'][i])**tetherExp
                      if k_vec[1] > kMax:
                          k_vec[1] = kMax
                      position[0] = restraints[str(genome_locus)+'_muR'][i]*R
                      k_vec[0] = tetherAlphaR/(restraints[str(genome_locus)+'_sigmaR'][i])**tetherExp
                      if k_vec[0] > kMax:
                          k_vec[0] = kMax
                  else:
                      position[0] = 0.2*R
                      k_vec[0] = tetherAlphaRnoInput
                      tmpP = particles[np.logical_and(particles>=N/4, particles<2*N/4)]
                      nearP1 = tmpP[tmpP<p]
                      if nearP1.any():
                          nearP1 = nearP1[-1]
                          nearG1 = genome_loci[bead_num==nearP1-int(N/4)][0]
                          pos1 = restraints[str(nearG1)+'_mu2'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(nearG1)+'_sigma2'][i]
                      else:
                          nearP1 = tmpP[-1]
                          nearG1 = genome_loci[bead_num==nearP1-int(N/4)][0]-N_MG1655
                          pos1 = restraints[str(genome_loci[bead_num==nearP1-int(N/4)][0])+'_mu2'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(genome_loci[bead_num==nearP1-int(N/4)][0])+'_sigma2'][i]
                          
                      nearP2 = tmpP[tmpP>p]
                      if nearP2.any():
                          nearP2 = nearP2[0]
                          nearG2 = genome_loci[bead_num==nearP2-int(N/4)][0]
                          pos2 = restraints[str(nearG2)+'_mu2'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(nearG2)+'_sigma2'][i]
                      else:
                          nearP2 = tmpP[0]
                          nearG2 = N_MG1655+genome_loci[bead_num==nearP2-int(N/4)][0]
                          pos2 = restraints[str(genome_loci[bead_num==nearP2-int(N/4)][0])+'_mu2'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(genome_loci[bead_num==nearP2-int(N/4)][0])+'_sigma2'][i]
                      
                      g=grid[p-int(N/4)]
                      m = (pos2-pos1)/(nearG2-nearG1)
                      b = pos1-m*nearG1
                      pos = m*g + b
                      s = (s1+s2)/2
                      position[1] = pos
                      k_vec[1] = tetherAlpha/s**tetherExp
              
              elif p>=2*N/4 and p<3*N/4:
                  if p in particles:
                      genome_locus = genome_loci[bead_num==p-2*int(N/4)][0]
                      position[1] = restraints[str(genome_locus)+'_mu3'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                      k_vec[1] = tetherAlpha/(restraints[str(genome_locus)+'_sigma3'][i])**tetherExp
                      if k_vec[1] > kMax:
                          k_vec[1] = kMax
                      position[0] = restraints[str(genome_locus)+'_muR'][i]*R
                      k_vec[0] = tetherAlphaR/(restraints[str(genome_locus)+'_sigmaR'][i])**tetherExp
                      if k_vec[0] > kMax:
                          k_vec[0] = kMax
                  else:
                      position[0] = 0.2*R
                      k_vec[0] = tetherAlphaRnoInput
                      tmpP = particles[np.logical_and(particles>=2*N/4, particles<3*N/4)]
                      nearP1 = tmpP[tmpP<p]
                      if nearP1.any():
                          nearP1 = nearP1[-1]
                          nearG1 = genome_loci[bead_num==nearP1-2*int(N/4)][0]
                          pos1 = restraints[str(nearG1)+'_mu3'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(nearG1)+'_sigma3'][i]
                      else:
                          nearP1 = tmpP[-1]
                          nearG1 = genome_loci[bead_num==nearP1-2*int(N/4)][0]-N_MG1655
                          pos1 = restraints[str(genome_loci[bead_num==nearP1-2*int(N/4)][0])+'_mu3'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(genome_loci[bead_num==nearP1-2*int(N/4)][0])+'_sigma3'][i]
                          
                      nearP2 = tmpP[tmpP>p]
                      if nearP2.any():
                          nearP2 = nearP2[0]
                          nearG2 = genome_loci[bead_num==nearP2-2*int(N/4)][0]
                          pos2 = restraints[str(nearG2)+'_mu3'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(nearG2)+'_sigma3'][i]
                      else:
                          nearP2 = tmpP[0]
                          nearG2 = N_MG1655+genome_loci[bead_num==nearP2-2*int(N/4)][0]
                          pos2 = restraints[str(genome_loci[bead_num==nearP2-2*int(N/4)][0])+'_mu3'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(genome_loci[bead_num==nearP2-2*int(N/4)][0])+'_sigma3'][i]
                      
                      g=grid[p-2*int(N/4)]
                      m = (pos2-pos1)/(nearG2-nearG1)
                      b = pos1-m*nearG1
                      pos = m*g + b
                      s = (s1+s2)/2
                      position[1] = pos
                      k_vec[1] = tetherAlpha/s**tetherExp
                      
              
              elif p>=3*N/4:
                  if p in particles:
                      genome_locus = genome_loci[bead_num==p-3*int(N/4)][0]
                      position[1] = restraints[str(genome_locus)+'_mu4'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                      k_vec[1] = tetherAlpha/(restraints[str(genome_locus)+'_sigma4'][i])**tetherExp
                      if k_vec[1] > kMax:
                          k_vec[1] = kMax
                      position[0] = restraints[str(genome_locus)+'_muR'][i]*R
                      k_vec[0] = tetherAlphaR/(restraints[str(genome_locus)+'_sigmaR'][i])**tetherExp
                      if k_vec[0] > kMax:
                          k_vec[0] = kMax
                  else:
                      position[0] = 0.2*R
                      k_vec[0] = tetherAlphaRnoInput
                      tmpP = particles[particles>=3*N/4]
                      nearP1 = tmpP[tmpP<p]
                      if nearP1.any():
                          nearP1 = nearP1[-1]
                          nearG1 = genome_loci[bead_num==nearP1-3*int(N/4)][0]
                          pos1 = restraints[str(nearG1)+'_mu4'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(nearG1)+'_sigma4'][i]
                      else:
                          nearP1 = tmpP[-1]
                          nearG1 = genome_loci[bead_num==nearP1-3*int(N/4)][0]-N_MG1655
                          pos1 = restraints[str(genome_loci[bead_num==nearP1-3*int(N/4)][0])+'_mu4'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s1 = restraints[str(genome_loci[bead_num==nearP1-3*int(N/4)][0])+'_sigma4'][i]
                          
                      nearP2 = tmpP[tmpP>p]
                      if nearP2.any():
                          nearP2 = nearP2[0]
                          nearG2 = genome_loci[bead_num==nearP2-3*int(N/4)][0]
                          pos2 = restraints[str(nearG2)+'_mu4'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(nearG2)+'_sigma4'][i]
                      else:
                          nearP2 = tmpP[0]
                          nearG2 = N_MG1655+genome_loci[bead_num==nearP2-3*int(N/4)][0]
                          pos2 = restraints[str(genome_loci[bead_num==nearP2-3*int(N/4)][0])+'_mu4'][i]*restraints.cell_length[i]*L/restraints.cell_length[0]
                          s2 = restraints[str(genome_loci[bead_num==nearP2-3*int(N/4)][0])+'_sigma4'][i]
                          
                      g=grid[p-3*int(N/4)]
                      m = (pos2-pos1)/(nearG2-nearG1)
                      b = pos1-m*nearG1
                      pos = m*g + b
                      s = (s1+s2)/2
                      position[1] = pos
                      k_vec[1] = tetherAlpha/s**tetherExp
                      
              local_force = forces.tether_particles_radial(sim, [p], k=k_vec, positions=position, name="tether_%d" % p)
              sim.add_force(local_force)
              position_vec[:,p] = position
              k_vec_all[:,p] = k_vec
              
      # plt.plot(range(N),k_vec_all[0])
      # plt.savefig('debugconstraints.png')
      # return
  
      replication_fork_bonds = []
      if len(segregated_sites_2)>0:
          replication_fork_bonds.append([segregated_array_2[0], int(np.mod(segregated_array_2[0]-N/4-1,N/4))])
          replication_fork_bonds.append([segregated_array_2[-1],int(np.mod(segregated_array_2[-1]-N/4+1,N/4))])
          
      if len(segregated_sites_3)>0:
          replication_fork_bonds.append([segregated_array_3[0], int(np.mod(segregated_array_3[0]-2*N/4-1,N/4))])
          replication_fork_bonds.append([segregated_array_3[-1],int(np.mod(segregated_array_3[-1]-2*N/4+1,N/4))])
      
      if len(segregated_sites_4)>0:
          replication_fork_bonds.append([segregated_array_4[0], int(np.mod(segregated_array_4[0]-3*N/4-1,N/4))+int(N/4)])
          replication_fork_bonds.append([segregated_array_4[-1],int(np.mod(segregated_array_4[-1]-3*N/4+1,N/4))+int(N/4)])
          
      sim.add_force(
          forcekits.polymer_chains(
              sim,
              # chains=[(0, None, False)],
              chains=[(0, int(N/4),True), (int(N/4), 2*int(N/4), True), (2*int(N/4), 3*int(N/4), True), (3*int(N/4), None, True)],
              # By default the library assumes you have one polymer chain
              # If you want to make it a ring, or more than one chain, use self.setChains
              # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from monomer 50 to the end
              bond_force_func=forces.harmonic_bonds,
              bond_force_kwargs={
                  "bondLength": 1.0,
                  "bondWiggleDistance": 0.05,  # Bond distance will fluctuate +- 0.05 on average
              },
              angle_force_func=forces.angle_force,
              angle_force_kwargs={
                  "k": angK,
                  # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                  # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
              },
              nonbonded_force_func=forces.polynomial_repulsive_with_exclusions,
              nonbonded_force_kwargs={
                  #"trunc": 0,
                  # "trunc": 3.0,  # this will let chains cross sometimes
                  # 'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
                  "trunc": trunc_array,
              },
              except_bonds=True,
              # extra_bonds = phantom_particles_bonds
              extra_bonds = replication_fork_bonds,
          )
      )
      
      sim.local_energy_minimization()
      if i==0:
        sim.do_block(200000) # equilibirate polymer before starting the simulation
       
      for j in range(blocks_per_bin):  # Do "blocks_per_bin" blocks
          print("running instance %d/%d size bin %d/%d, block %d/%d" % (n, args.num_instances, i,num_of_bins,j,blocks_per_bin))
          sim.do_block(int(steps_per_bin/blocks_per_bin))   
      sim.print_stats()  # In the end, print very simple statistics
      print("finished bin #%d" % i)
      clean_polychrom_folder(instance_folder)
      
      f = open(instance_folder + "/log.txt", "a")
      f.write("finished bin #%d" % i)
      f.close()

  #%%.cs
  reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
  np.savetxt(instance_folder + "/massArray_bins.csv",mass_array_bins,delimiter=',')

parser=create_parser()
args = parser.parse_args()


if args.num_nodes>1:
    assert args.node_ind, "When number of nodes is >1 node_ind must be set"
    assert args.node_ind<=args.num_nodes, "node_ind must be <= number of nodes"
    time.sleep(args.node_ind) # wait a second to prevent collision between nodes
    
start_time = time.time()
folder = args.output_folder_prefix
folder = 'Ensemble_models/' + folder
if args.no_restraints:
    folder = folder + '_noRestraints'
    
if args.no_radial:
    folder = folder + '_noRadial'
    
if args.randomize_radial:
    folder = folder + '_randomRadial'
    print('seed = %d\n' % args.seed)
    
if args.randomize_all:
    folder = folder + '_randomAll_%d' % args.seed
    print('seed = %d\n' % args.seed)
    
if args.leave_out_all_ind >= 0:
    folder = folder + '_%d' % args.leave_out_all_ind
    
if not os.path.isdir(folder):
    os.mkdir(folder)


instances_list = glob.glob(folder + '/*/')

if args.num_nodes==1:
    start_instance = 0
    end_instance = args.num_instances
else:
    start_instance = int(np.floor((args.node_ind-1)*(args.num_instances/args.num_nodes)))
    end_instance = int(np.floor((args.node_ind)*(args.num_instances/args.num_nodes)))

if len(instances_list)>0:
    instances_list.sort()
    if args.num_nodes==1:
        last_instance = -1
        while (last_instance==-1):
            try: 
                list_URIs(instances_list[-1])
                last_instance = int(instances_list[-1][-3:])
            except:
                if len(instances_list)>0:
                    instances_list.pop(-1)
                    pass
                else:
                    last_instance = -1
                    break
    else:
        instances_inds = np.array(list(map(lambda x: int(x[-4:-1]),instances_list)))
        instances_inds = np.where(np.logical_and(instances_inds>=start_instance,instances_inds<end_instance))[0]
        if len(instances_inds)>0:
            instances_list = instances_list[instances_inds[0]:instances_inds[-1]+1]
            last_instance = start_instance-1
            while (last_instance==start_instance-1):
                try: 
                    list_URIs(instances_list[-1])
                    last_instance = int(instances_list[-1][-3:])
                except:
                    if len(instances_list)>0:
                        instances_list.pop(-1)
                        pass
                    else:
                        last_instance = start_instance-1
                        break
        else:
            last_instance = start_instance-1
            
            
else:
    last_instance = start_instance-1

if args.num_nodes > 1:
    node_ind = args.node_ind
else:
    node_ind = 1

print ('ind = %d' % node_ind)

for i in range(last_instance+1,end_instance):
    flag = True
    #run_sim(i, folder, args)
    while flag:
        try:
          run_sim(i, folder, args)
          flag = False
        except:
          pass
# with Pool(cpu_count()) as p:
#     p.map(run_sim, range(100))
print("execution time = %f seconds" % (time.time() - start_time))
