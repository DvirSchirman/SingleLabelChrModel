#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:42:24 2024

@author: dvirs
"""

import glob
import os

folder = '241023_forkPlots_dropped_inputs_randomModel'
forkFiles = glob.glob('%s/simulatedForkPlots*/' % folder)

if not os.path.isdir('%s/ForkMatFiles/' % folder):
    os.mkdir('%s/ForkMatFiles/' % folder)
if not os.path.isdir('%s/RadialMatFiles/' % folder):
    os.mkdir('%s/RadialMatFiles/' % folder)

for f in forkFiles:
    os.system('mv %snonRestrainedStrains/*.mat %s/ForkMatFiles/' % (f, folder))

radialFiles = glob.glob(folder + '/simulatedRadialPlots*/')
for f in radialFiles:
    os.system('mv %snonRestrainedStrains/*.mat %s/RadialMatFiles/' % (f, folder))