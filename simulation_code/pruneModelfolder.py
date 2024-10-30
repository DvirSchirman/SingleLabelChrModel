#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:51:39 2024

@author: dvirs
"""

import glob
import shutil

modelFolder = 'randomModels'
numInstances = 10

subfolders = glob.glob(modelFolder + '/*') 

for f in subfolders:
    count = 0
    rmList = []
    modelInstances = glob.glob(f + '/*')
    for m in modelInstances:
        if count<numInstances:
            files = glob.glob(m + '/*')
            if len(files)==3:
                count+=1
            else:
                rmList.append(m)
        else:
            rmList.append(m)
    for m in rmList:
        shutil.rmtree(m)