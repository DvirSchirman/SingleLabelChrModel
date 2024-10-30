# Introduction
This repository include the simulation and analysis code for the data-driven chromosome polymer model dexcrined in the paper named **"A dynamic 3D polymer model of the Escherichia coli chromosome driven by data from optical pooled screening"**. 
It also includes some pooled library imaging results visualization scirpts.

# Data-driven chromosome polymer model
## Code dependencies
- The simulation code is based on Polychrom (v0.1.0) https://github.com/open2c/polychrom
- First, after downloading Polychrom, replace 3 of Polychrom source files with updated files from this repository. copy the source files from this reposiory simulation_code/polychrom_files/ to polychrom/polychrom/
- Second, install Polychrom according to instructions https://polychrom.readthedocs.io/en/latest/. Notice that this simulation doesn't use GPU so there is no need to install CUDA.
- More Python packages used for simulation or analysis:
  - mdanalysis
  - matplotlib
  - seaborn
  - palettable
 

## Run the simulation
For running the simulation run simulation_code/localization_based_model_constrained.py.<br>
Run `python localization_based_model_constrained.py --help` for a description of the different parameters.<be>
A typical simulation instance run time on Uppsala University's high-performance computing cluster was around 6 hours.
To run an ensemble of independent simulation instances on a cluster using Slurm you can run the bash script simulation_code/run_sim_localization_based_model_array_cpu.sh
