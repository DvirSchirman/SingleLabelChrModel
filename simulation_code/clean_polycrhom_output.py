#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 11:59:28 2023

@author: dvirs
"""

import os
import glob

def clean_polychrom_folder(folder):
    file_list = glob.glob(folder + '/*')
    # print(len(file_list))
    file_list = [i for i in file_list if 'blocks' not in i]
    file_list = [i for i in file_list if 'massArray' not in i]
    # print(len(file_list))
    for f in file_list:
        os.remove(f)
    
folder = "full_cell_cycle_model_200k_2D_repInput_pooledExp_angK4_005"

clean_polychrom_folder(folder)

# subfolders = glob.glob(folder + '/*')
# for s in subfolders:
#     clean_polychrom_folder(s)