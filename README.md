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
For running the simulation run `simulation_code/localization_based_model_constrained.py`.<br>
Run `python localization_based_model_constrained.py --help` for a description of the different parameters.<be>
A typical simulation instance run time on Uppsala University's high-performance computing cluster was around 6 hours.
To run an ensemble of independent simulation instances on a cluster using Slurm you can run the bash script `simulation_code/run_sim_localization_based_model_array_cpu.sh`

## Simulation analysis
### Pre-ran simulations
Pre-ran simulations' raw data used for this paper, available for further analysis can be found [here](https://figshare.com/ndownloader/files/50078703?private_link=ef5f1fbb7a4a4daf0b6a)

### Get simulated localization distribution plots
- To get simulated distribution plots for each library locus (Fig. 5b) run `simulation_code/getForkPlots.py`.
- In the case of an ensemble with dropped-out inputs, if you wish to analyze only the dropped-out loci you can run `simulation_code/getForkPlotsDroppedLoci.py`

### Get pairwise distance matrices
- Bulk pairwise distance matrices
  - To get a bulk pairwise distance matrix for **all distances**, run first `simulation_code/getBulkDistanceMat.py`, and then `simulation_code/plotBulkDistanceMap.py` (Fig. 6a)
  - To get a bulk pairwise distance matrix for **interchromosomal & intrachromosomal distances**, run first `simulation_code/getBulkDistanceMat_CisTrans.py`, and then `simulation_code/plotBulkDistanceMap.py --CisTrans` (Fig. 6c)
- Dynamic pairwise distance matrices
  - To get distance matrices per each cell size bin  for **all distances**, run first `simulation_code/getDynamicDistanceMat.py`, and then `simulation_code/plotDynamicDistanceMap.py` (Fig. 6d)
  - To get distance matrices per each cell size bin  for **interchromosomal & intrachromosomal distances**, run first `simulation_code/getDynamicDistanceMat_CisTrans.py`, and then `simulation_code/plotDynamicDistanceMap.py --CisTrans` (Fig. 6e)
 
### Visualize the model output
You can visualize the model's output using [PyMol](https://pymol.org/) (Movie S3).<br>
To visualize the dynamic model output, first convert the model's binary output data to a .pdb trajectory file using `simulation_code/convert_trj2pdb.py`. Then open PyMol and within it run `simulation_code/visualize_trj_restraints_repInput.py` 

# Imaging data visualization
Jupyter notebook `visualize_data/Visualize_library_data.ipynb` can be used to reproduce the following: 
- Library compositions maps presented in figures 1a & 2
- Figures 3, 4, & S7
- Movies S1, S2
## Code dependencies
The following python packages are needed:
- biopython
- pycirclize
- pandas
- numpy
- matplotlib
- seaborn
