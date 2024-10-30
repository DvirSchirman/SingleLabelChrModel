import pymol
from pymol.cgo import *
#%%
import math

import random
import numpy as np
import pandas as pd
from palettable.tableau import Tableau_20 as Color_set
from palettable.colorbrewer.qualitative import Dark2_3 as Color_set2

def cgoCircle(x, y, z, r=8.0, cr=1.0, cg=0.4, cb=0.8, w=2.0):
  """
  Create a CGO circle

  PARAMS
        x, y, z
          X, Y and Z coordinates of the origin

        r
          Radius of the circle

        cr, cg, cb
          Color triplet, [r,g,b] where r,g,b are all [0.0,1.0].

        w
          Line width of the circle

  RETURNS
        the CGO object (it also loads it into PyMOL, too).

  """
  x = float(x)
  y = float(y)
  z = float(z)
  r = abs(float(r))
  cr = abs(float(cr))
  cg = abs(float(cg))
  cb = abs(float(cb))
  w = float(w)

  obj = [ BEGIN, LINES, COLOR, cr, cg, cb ]
  print(z)
  for i in range(180):
        obj.append( VERTEX )
        obj.append(r*math.cos(i) + x )
        obj.append(r*math.sin(i) + y )
        obj.append(z)
        obj.append( VERTEX )
        obj.append(r*math.cos(i+0.1) + x )
        obj.append(r*math.sin(i+0.1) + y )
        obj.append(z)
  obj.append(END)
 
  # print(obj) 
  cName = cmd.get_unused_name("circle_")
  cmd.load_cgo( obj, cName )
  cmd.set("cgo_line_width", w, cName )
  return obj

# def main():
cmd.do("reinitialize")

# ~ cmd.do("set matrix_mode, 1")
cmd.do("set movie_panel, 1")
cmd.do("set scene_buttons, 1")
cmd.do("set cache_frames, 1")

# cmd.load('test.pdb')
cmd.load('pdbs/241022_4Paper_normalModel.pdb')
N_MG1655 = 4641652
oriC_site = 3925860
N = 1000 # TODO: read N from data (when replicated number of atoms is at the range [N,N*2])

cmd.extend( "cgoCircle", cgoCircle )

volume_ratio = 0.1
r_monomer = 0.5 #never change
v_monomer = (4/3)*np.pi*r_monomer**3 #calculating volume
v_cylinder = v_monomer*N*(1/volume_ratio) #total volume of the cylinder=cell
L=(4*v_cylinder/np.pi)**(1/3) #length of cylinder (DS 230516: changed 2V to 4V)
R=L/2 #radius of cylinder
L = round(L,1)
R = round(R,1)

cmd.do('hide everything, all')
cmd.do('show cartoon, all')
cmd.do('set cartoon_sampling, 2')
cmd.do('set cartoon_tube_radius, 0.35')

# cmd.do('hide everything, all')
# cmd.do('show cartoon, all')

# cmd.do('toggle everything, chain F')
# ~ cmd.do('color 0xFFFFFF')
cmd.do('color 0xbe2931')
# cmd.do('color 0x777777, (chain B)')
c = Color_set.hex_colors[0][1:]
cmd.do('color 0x%s, (chain B)' % c)
c = Color_set.hex_colors[13][1:]
cmd.do('color 0x%s, (chain C)' % c)
c = Color_set.hex_colors[1][1:]
cmd.do('color 0x%s, (chain D)' % c)
c = Color_set.hex_colors[12][1:]
cmd.do('color 0x%s, (chain E)' % c)

num_of_frames = cmd.count_states()

# cmd.set_view ([\
#     -0.118596636,    0.573426187,   -0.810628474,\
#      0.063117012,    0.819092214,    0.570179224,\
#      0.990935445,    0.016456395,   -0.133334696,\
#      0.000000000,    0.000000000, -381.041412354,\
#     -0.476818085,    0.123840332,    0.158771515,\
#    300.415954590,  461.666870117,  -20.000000000] )
    
cmd.set_view ([\
     -0.006043129,    0.880124688,    0.474704236,\
     0.065737888,   -0.473335743,    0.878425717,\
     0.997820258,    0.036514848,   -0.054997496,\
     0.000000000,    0.000000000, -136.174545288,\
    -0.094497681,    0.059288979,    0.671245575,\
   107.361053467,  164.988037109,  -20.000000000] )

cmd.do("mset %d, %d" % (1,1))
cmd.do("frame %d" % 1)
circle1 = cgoCircle(0,0,0-L/2,R, 1, 1, 1)
circle2 = cgoCircle(0,0,0+L/2,R, 1, 1, 1)
# cmd.do("mview store")
cmd.do("mview store, object=circle1")
cmd.do("mview store, object=circle2")

# localization_restraints_file = '../data/230615_localization_data_radial.csv'
localization_restraints_file = '../data/240528_localization_data_radial_repInput_normalizedR_polyfit.csv'
restraints = pd.read_csv(localization_restraints_file,sep = ',')
num_of_bins = len(restraints)
genome_loci = restraints.columns[1:]
genome_loci = [int(a.split(sep='_')[0]) for a in genome_loci]
genome_loci = np.unique(genome_loci)
grid = np.linspace(1,N_MG1655,N,dtype=int)
bead_num = np.array([np.argmin(abs(grid-i)) for i in genome_loci])
oriC_bead_num = np.argmin(abs(grid-oriC_site))

if len(genome_loci)>10:
    inds = inds = np.random.choice(range(len(genome_loci)),10,replace=0)
    genome_loci = genome_loci[inds]
    bead_num = bead_num[inds]

# ~ position1 = np.zeros(len(genome_loci))
# ~ position2 = np.zeros(len(genome_loci))
# ~ for i,g in enumerate(genome_loci):
    # ~ position1[i] = restraints[str(g)+'_mu1'][0]*L
    # ~ position2[i] = restraints[str(g)+'_mu2'][0]*L
    # ~ c1 = Color_set.colors[2*i]
    # ~ c2 = Color_set.colors[2*i+1]
    # ~ cgoCircle(0,0,0+position1[i],R,c1[0]/255, c1[1]/255, c1[2]/255)
    # ~ if np.isfinite(position2[i]):
        # ~ cgoCircle(0,0,0+position2[i],R,c2[0]/255, c2[1]/255, c2[2]/255)
    # ~ else:
        # ~ cgoCircle(0,0,0,R,c2[0]/255, c2[1]/255, c2[2]/255)
    # ~ string = 'color 0x' + Color_set.hex_colors[2*i][1:] + ', resi ' + str(bead_num[i])
    # ~ cmd.do(string)
    # ~ cmd.do("toggle sphere, resi " + str(bead_num[i]))
    # ~ string = 'color 0x' + Color_set.hex_colors[2*i+1][1:] + ', resi ' + str(bead_num[i]+N)
    # ~ cmd.do(string)
    # ~ cmd.do("toggle sphere, resi " + str(bead_num[i]+N))
    
frames_per_bin = num_of_frames/num_of_bins
    
n=1
for i in range(1,num_of_frames):
    # cmd.do("mset %d, %d" % (i,i))
    # cmd.do("frame %d" % i)
    for f in range(2):
        cmd.do("mset %d, %d" % (i,n))
        n+=1
    cmd.do("frame %d" % n)
    if not np.mod(i,frames_per_bin):
        bin = int(i/frames_per_bin)
        print("bin = %d\n" % bin)
        added_length = (restraints.cell_length[bin]-restraints.cell_length[bin-1])/restraints.cell_length[0]*L
        cmd.do("translate [%f,0,0], object=circle_01" % (-added_length/2))
        cmd.do("translate [%f,0,0], object=circle_02" % (added_length/2))
        ##cmd.do("mview store")
        cmd.do("mview store, object=circle_01")
        cmd.do("mview store, object=circle_02")
        # ~ for j,g in enumerate(genome_loci):
            # ~ prev_pos = position1[j]
            # ~ position1[j] = restraints[str(g)+'_mu1'][bin]*restraints.cell_length[bin]*L/restraints.cell_length[0]
            # ~ dpos = position1[j]-prev_pos
            # ~ cmd.do("translate [%f,0,0], object=circle_%2d" % (dpos, 2*j+3))
            # ~ prev_pos = position2[j]
            # ~ position2[j] = restraints[str(g)+'_mu2'][bin]*restraints.cell_length[bin]*L/restraints.cell_length[0]
            # ~ if np.isfinite(position2[j]):             
                # ~ if np.isfinite(prev_pos):
                    # ~ dpos = position2[j]-prev_pos
                # ~ else:
                    # ~ dpos = position2[j]
                # ~ print("dpos2 = %1.3f, state=%d, genome_locus=%d" % (dpos, i, g))
                # ~ cmd.do("translate [%f,0,0], object=circle_%2d" % (dpos, 2*j+4))
            # ~ cmd.do("mview store, object=circle_%2d" % (2*j+3) )
            # ~ cmd.do("mview store, object=circle_%2d" % (2*j+4))
        # cmd.do("mview store")
    
    print(i)
    print(num_of_frames)
 
cmd.do('zoom all') 
cmd.do("frame 1")
cmd.mplay
    
    
    
