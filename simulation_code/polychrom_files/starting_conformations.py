import warnings
from math import sqrt, sin, cos

import numpy as np
import pandas as pd
from numpy import random

def create_spiral(r1, r2, N):
    """
    Creates a "propagating spiral", often used as an easy mitotic-like
    starting conformation.

    Run it with r1=10, r2 = 13, N=5000, and see what it does.
    """
    Pi = 3.141592
    points = []
    finished = [False]

    def rad(phi):
        return phi / (2 * Pi)

    def ang(rad):
        return 2 * Pi * rad

    def coord(phi):
        r = rad(phi)
        return r * sin(phi), r * cos(phi)

    def fullcoord(phi, z):
        c = coord(phi)
        return [c[0], c[1], z]

    def dist(phi1, phi2):
        c1 = coord(phi1)
        c2 = coord(phi2)
        d = sqrt((c1[1] - c2[1]) ** 2 + (c1[0] - c2[0]) ** 2)
        return d

    def nextphi(phi):
        phi1 = phi
        phi2 = phi + 0.7 * Pi
        mid = phi2
        while abs(dist(phi, mid) - 1) > 0.00001:
            mid = (phi1 + phi2) / 2.0
            if dist(phi, mid) > 1:
                phi2 = mid
            else:
                phi1 = mid
        return mid

    def prevphi(phi):

        phi1 = phi
        phi2 = phi - 0.7 * Pi
        mid = phi2

        while abs(dist(phi, mid) - 1) > 0.00001:
            mid = (phi1 + phi2) / 2.0
            if dist(phi, mid) > 1:
                phi2 = mid
            else:
                phi1 = mid
        return mid

    def add_point(point, points=points, finished=finished):
        if (len(points) == N) or (finished[0] == True):
            points = np.array(points)
            finished[0] = True
            print("finished!!!")
        else:
            points.append(point)

    z = 0
    forward = True
    curphi = ang(r1)
    add_point(fullcoord(curphi, z))
    while True:
        if finished[0] == True:
            return np.array(points)
        if forward == True:
            curphi = nextphi(curphi)
            add_point(fullcoord(curphi, z))
            if rad(curphi) > r2:
                forward = False
                z += 1
                add_point(fullcoord(curphi, z))
        else:
            curphi = prevphi(curphi)
            add_point(fullcoord(curphi, z))
            if rad(curphi) < r1:
                forward = True
                z += 1
                add_point(fullcoord(curphi, z))


def _random_points_sphere(N):
    theta = np.random.uniform(0.0, 1.0, N)
    theta = 2.0 * np.pi * theta

    u = np.random.uniform(0.0, 1.0, N)
    u = 2.0 * u - 1.0
    
    return np.vstack([theta, u]).T
    
    
def create_random_walk(step_size, N):
    """
    Creates a freely joined chain of length N with step step_size 
    """
    
    theta, u =  _random_points_sphere(N).T
    
    dx = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
    dy = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
    dz = step_size * u    
    
    x, y, z = np.cumsum(dx), np.cumsum(dy), np.cumsum(dz)
        
    return np.vstack([x, y, z]).T


def create_constrained_random_walk(N, 
    constraint_f, 
    starting_point = (0, 0, 0),
    step_size=1.0
    ):
    """
    Creates a constrained freely joined chain of length N with step step_size.
    Each step of a random walk is tested with the constraint function and is
    rejected if the tried step lies outside of the constraint.
    This function is much less efficient than create_random_walk().
   
    Parameters
    ----------
    N : int
        The number of steps
    constraint_f : function((float, float, float))
        The constraint function. 
        Must accept a tuple of 3 floats with the tentative position of a particle
        and return True if the new position is accepted and False is it is forbidden.
    starting_point : a tuple of (float, float, float)
        The starting point of a random walk.
    step_size: float
        The size of a step of the random walk.

    """    
    
    i = 1
    j = N
    out = np.full((N, 3), np.nan)
    out[0] = starting_point
    
    while i < N:
        if j == N:
            theta, u = _random_points_sphere(N).T        
            dx = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
            dy = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
            dz = step_size * u
            d = np.vstack([dx, dy, dz]).T
            j = 0
    
        new_p = out[i-1] + d[j]
        
        if constraint_f(new_p):
            out[i] = new_p
            i += 1
        
        j += 1
        
    return out


def grow_cubic(N, boxSize, method="standard"):
    """
    This function grows a ring or linear polymer on a cubic lattice 
    in the cubic box of size boxSize. 
    
    If method=="standard, grows a ring starting with a 4-monomer ring in the middle 
    
    if method =="extended", it grows a ring starting with a long ring 
    going from z=0, center of XY face, to z=boxSize center of XY face, and back. 
    
    If method="linear", then it grows a linearly organized chain from 0 to size.
    The chain may stick out of the box by one, (N%2 != boxSize%2), or be flush with the box otherwise

    Parameters
    ----------
    N: chain length. Must be even for rings. 
    boxSize: size of a box where polymer is generated.
    method: "standard", "linear" or "extended"


    """
    if N > boxSize ** 3:
        raise ValueError("Steps ahs to be less than size^3")
    if N > 0.9 * boxSize ** 3:
        warnings.warn("N > 0.9 * boxSize**3. It will be slow")
    if (N % 2 != 0) and (method != "linear"):
        raise ValueError("N has to be multiple of 2 for rings")

    t = boxSize // 2
    if method == "standard":
        a = [(t, t, t), (t, t, t + 1), (t, t + 1, t + 1), (t, t + 1, t)]

    elif method == "extended":
        a = []
        for i in range(1, boxSize):
            a.append((t, t, i))

        for i in range(boxSize - 1, 0, -1):
            a.append((t, t - 1, i))
        if len(a) > N:
            raise ValueError("polymer too short for the box size")

    elif method == "linear":
        a = []
        for i in range(0, boxSize + 1):
            a.append((t, t, i))
        if (len(a) % 2) != (N % 2):
            a = a[1:]
        if len(a) > N:
            raise ValueError("polymer too short for the box size")

    else:
        raise ValueError("select methon from standard, extended, or linear")

    b = np.zeros((boxSize + 2, boxSize + 2, boxSize + 2), int)
    for i in a:
        b[i] = 1

    for i in range((N - len(a)) // 2):
        while True:
            if method == "linear":
                t = np.random.randint(0, len(a) - 1)
            else:
                t = np.random.randint(0, len(a))

            if t != len(a) - 1:
                c = np.abs(np.array(a[t]) - np.array(a[t + 1]))
                t0 = np.array(a[t])
                t1 = np.array(a[t + 1])
            else:
                c = np.abs(np.array(a[t]) - np.array(a[0]))
                t0 = np.array(a[t])
                t1 = np.array(a[0])
            cur_direction = np.argmax(c)
            while True:
                direction = np.random.randint(0, 3)
                if direction != cur_direction:
                    break
            if np.random.random() > 0.5:
                shift = 1
            else:
                shift = -1
            shiftar = np.array([0, 0, 0])
            shiftar[direction] = shift
            t3 = t0 + shiftar
            t4 = t1 + shiftar
            if (
                (b[tuple(t3)] == 0)
                and (b[tuple(t4)] == 0)
                and (np.min(t3) >= 1)
                and (np.min(t4) >= 1)
                and (np.max(t3) < boxSize + 1)
                and (np.max(t4) < boxSize + 1)
            ):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a
    return np.array(a) - 1

def restrained_rect_box_1D(N,boxSize,restraints):
    boxSize = np.array(boxSize)
    if N > boxSize.prod():
        raise ValueError("Steps has to be less than volume of box")
    if N > 0.9 * boxSize.prod():
        warnings.warn("N > 0.9 * box volume. It will be slow")
        
    # restraints = pd.read_csv(restraintFile,sep = ',')
    if restraints.bead_num.max() > N:
        raise ValueError("restraints bead numbers exceed N")
        
    tx = boxSize[0] // 2
    ty = boxSize[1] // 2
    tz = boxSize[2] // 2
    tmp = restraints.Z_mu.values*tz*2
    tmp = tmp.round()+tz*2
    restraints['Zgrid']= tmp

    restraints = restraints.sort_values('bead_num')

    a = -np.ones((N,3),int) 
    for i in range(len(restraints)):
        ind = restraints.iloc[i].bead_num
        a[ind,2] = restraints.iloc[i].Zgrid
        if restraints.iloc[i].side == 'F':
            a[ind,0] =  np.random.randint(tx,boxSize[0])
        else:
            a[ind,0] =  np.random.randint(0,tx)
        a[ind,1] = np.random.randint(0,boxSize[1])
        
        if i>0:
            # min_dist = abs(a[ind,0]-a[prev_ind,0])+abs(a[ind,1]-a[prev_ind,1])+abs(a[ind,2]-a[prev_ind,2])
            
            beads_num = restraints.iloc[i].bead_num - restraints.iloc[i-1].bead_num-1
            beads_num = int(beads_num)
            
            dim=0
            if a[prev_ind,dim]<a[ind,dim]:
                if (a[prev_ind,dim]-a[ind,dim])>=3:
                    x_vec = range(a[prev_ind,dim],a[ind,dim]+1)
                else:
                    if a[ind,dim]+2<boxSize[dim]:
                        x_vec = range(a[prev_ind,dim],a[ind,dim]+4)
                    else:
                        x_vec = range(a[prev_ind,dim]-3,a[ind,dim]+1)
            else:
                if (a[ind,dim]-a[prev_ind,dim])>=3:
                    x_vec = range(a[prev_ind,dim],a[ind,dim]-1,-1)
                else:
                    if a[ind,dim]-3>0:
                        x_vec = range(a[prev_ind,dim],a[ind,dim]-4,-1)
                    else:
                        x_vec = range(a[prev_ind,dim]+3,a[ind,dim]-1,-1)        
                
            dim=1
            if a[prev_ind,dim]<a[ind,dim]:
                if (a[prev_ind,dim]-a[ind,dim])>=3:
                    y_vec = range(a[prev_ind,dim],a[ind,dim]+1)
                else:
                    if a[ind,dim]+2<boxSize[dim]:
                        y_vec = range(a[prev_ind,dim],a[ind,dim]+4)
                    else:
                        y_vec = range(a[prev_ind,dim]-3,a[ind,dim]+1)
            else:
                if (a[ind,dim]-a[prev_ind,dim])>=3:
                    y_vec = range(a[prev_ind,dim],a[ind,dim]-1,-1)
                else:
                    if a[ind,dim]-3>0:
                        y_vec = range(a[prev_ind,dim],a[ind,dim]-4,-1)
                    else:
                        y_vec = range(a[prev_ind,dim]+3,a[ind,dim]-1,-1)        
            
            dim=2
            if a[prev_ind,dim]<a[ind,dim]:
                if (a[prev_ind,dim]-a[ind,dim])>=3:
                    z_vec = range(a[prev_ind,dim],a[ind,dim]+1)
                else:
                    if a[ind,dim]+2<boxSize[dim]:
                        z_vec = range(a[prev_ind,dim],a[ind,dim]+4)
                    else:
                        z_vec = range(a[prev_ind,dim]-3,a[ind,dim]+1)
            else:
                if (a[ind,dim]-a[prev_ind,dim])>=3:
                    z_vec = range(a[prev_ind,dim],a[ind,dim]-1,-1)
                else:
                    if a[ind,dim]-3>0:
                        z_vec = range(a[prev_ind,dim],a[ind,dim]-4,-1)
                    else:
                        z_vec = range(a[prev_ind,dim]+3,a[ind,dim]-1,-1)        
            
            grid_points = len(x_vec)*len(y_vec)*len(z_vec)
            grid_points = int(grid_points)
            tmp = -np.ones((grid_points,3),int)
            # print([grid_points, beads_num])
            # print(a[prev_ind])
            # print(a[ind])
            
            ii=0
            for iz in z_vec:
                for iy in y_vec:
                    for ix in x_vec:
                        tmp[ii,0]=ix
                        tmp[ii,1]=iy
                        tmp[ii,2]=iz
                        ii+=1
            tmp = np.delete(tmp,[0,-1],0)
            inds = random.choice(range(len(tmp)),beads_num,0)
            inds.sort()
            a[prev_ind+1:ind]=tmp[inds]
            
            if i==(len(restraints)-1):
                prev_ind = ind
                ind = restraints.iloc[0].bead_num
                
                beads_num = restraints.iloc[0].bead_num-1 + (N-restraints.iloc[i].bead_num)
                beads_num = int(beads_num)
                
                dim=0
                if a[prev_ind,dim]<a[ind,dim]:
                    if (a[prev_ind,dim]-a[ind,dim])>=2:
                        x_vec = range(a[prev_ind,dim],a[ind,dim]+1)
                    else:
                        if a[ind,dim]+1<boxSize[dim]:
                            x_vec = range(a[prev_ind,dim],a[ind,dim]+3)
                        else:
                            x_vec = range(a[prev_ind,dim]-2,a[ind,dim]+1)
                else:
                    if (a[ind,dim]-a[prev_ind,dim])>=2:
                        x_vec = range(a[prev_ind,dim],a[ind,dim]-1,-1)
                    else:
                        if a[ind,dim]-2>0:
                            x_vec = range(a[prev_ind,dim],a[ind,dim]-3,-1)
                        else:
                            x_vec = range(a[prev_ind,dim]+2,a[ind,dim]-1,-1)        
                    
                dim=1
                if a[prev_ind,dim]<a[ind,dim]:
                    if (a[prev_ind,dim]-a[ind,dim])>=2:
                        y_vec = range(a[prev_ind,dim],a[ind,dim]+1)
                    else:
                        if a[ind,dim]+1<boxSize[dim]:
                            y_vec = range(a[prev_ind,dim],a[ind,dim]+3)
                        else:
                            y_vec = range(a[prev_ind,dim]-2,a[ind,dim]+1)
                else:
                    if (a[ind,dim]-a[prev_ind,dim])>=2:
                        y_vec = range(a[prev_ind,dim],a[ind,dim]-1,-1)
                    else:
                        if a[ind,dim]-2>0:
                            y_vec = range(a[prev_ind,dim],a[ind,dim]-3,-1)
                        else:
                            y_vec = range(a[prev_ind,dim]+2,a[ind,dim]-1,-1)        
                
                dim=2
                if a[prev_ind,dim]<a[ind,dim]:
                    if (a[prev_ind,dim]-a[ind,dim])>=2:
                        z_vec = range(a[prev_ind,dim],a[ind,dim]+1)
                    else:
                        if a[ind,dim]+1<boxSize[dim]:
                            z_vec = range(a[prev_ind,dim],a[ind,dim]+3)
                        else:
                            z_vec = range(a[prev_ind,dim]-2,a[ind,dim]+1)
                else:
                    if (a[ind,dim]-a[prev_ind,dim])>=2:
                        z_vec = range(a[prev_ind,dim],a[ind,dim]-1,-1)
                    else:
                        if a[ind,dim]-2>0:
                            z_vec = range(a[prev_ind,dim],a[ind,dim]-3,-1)
                        else:
                            z_vec = range(a[prev_ind,dim]+2,a[ind,dim]-1,-1)        
                
                grid_points = len(x_vec)*len(y_vec)*len(z_vec)
                grid_points = int(grid_points)
                tmp = -np.ones((grid_points,3),int)
                
                ii=0
                for iz in z_vec:
                    for iy in y_vec:
                        for ix in x_vec:
                            tmp[ii,0]=ix
                            tmp[ii,1]=iy
                            tmp[ii,2]=iz
                            ii+=1
                tmp = np.delete(tmp,[0,-1],0)
                inds = random.choice(range(len(tmp)),beads_num,0)
                inds.sort()
                a[prev_ind+1:] = tmp[inds[0:len(a[prev_ind+1:])]]
                a[0:ind] = tmp[inds[len(a[prev_ind+1:]):]]
                
        prev_ind = ind
    return np.array(a) - 1

def grow_rect_box(N, boxSize, method="standard"):
    """
    This function grows a ring or linear polymer on a rectangle box lattice 
    boxSize - box size in [x,y,z] 
    
    If method=="standard, grows a ring starting with a 4-monomer ring in the middle 
    
    if method =="extended", it grows a ring starting with a long ring 
    going from z=0, center of XY face, to z=boxSize center of XY face, and back. 
    
    If method="linear", then it grows a linearly organized chain from 0 to size.
    The chain may stick out of the box by one, (N%2 != boxSize%2), or be flush with the box otherwise

    Parameters
    ----------
    N: chain length. Must be even for rings. 
    boxSize: box size vector in [x,y,z] 
    method: "standard", "linear" or "extended"


    """
    boxSize = np.array(boxSize)
    if N > boxSize.prod():
        raise ValueError("Steps has to be less than volume of box")
    if N > 0.9 * boxSize.prod():
        warnings.warn("N > 0.9 * box volume. It will be slow")
    if (N % 2 != 0) and (method != "linear"):
        raise ValueError("N has to be multiple of 2 for rings")

    # t = boxSize // 2
    tx = boxSize[0] // 2
    ty = boxSize[1] // 2
    tz = boxSize[2] // 2
    if method == "standard":
        a = [(tx, ty, tz), (tx, ty, tz+1), (tx, ty+1, tz+1), (tx, ty+1, tz)]

    elif method == "extended":
        a = []
        for i in range(1, boxSize[2]):
            a.append((tx, ty, i))

        for i in range(boxSize[2] - 1, 0, -1):
            a.append((tx, ty - 1, i))
        if len(a) > N:
            raise ValueError("polymer too short for the box size")

    elif method == "linear":
        a = []
        for i in range(0, boxSize + 1):
            a.append((tx, ty, i))
        if (len(a) % 2) != (N % 2):
            a = a[1:]
        if len(a) > N:
            raise ValueError("polymer too short for the box size")

    else:
        raise ValueError("select methon from standard, extended, or linear")

    b = np.zeros((boxSize[0] + 2, boxSize[1] + 2, boxSize[2] + 2), int)
    for i in a:
        b[i] = 1

    for i in range((N - len(a)) // 2):
        while True:
            if method == "linear":
                t = np.random.randint(0, len(a) - 1)
            else:
                t = np.random.randint(0, len(a))

            if t != len(a) - 1:
                c = np.abs(np.array(a[t]) - np.array(a[t + 1]))
                t0 = np.array(a[t])
                t1 = np.array(a[t + 1])
            else:
                c = np.abs(np.array(a[t]) - np.array(a[0]))
                t0 = np.array(a[t])
                t1 = np.array(a[0])
            cur_direction = np.argmax(c)
            while True:
                direction = np.random.randint(0, 3)
                if direction != cur_direction:
                    break
            if np.random.random() > 0.5:
                shift = 1
            else:
                shift = -1
            shiftar = np.array([0, 0, 0])
            shiftar[direction] = shift
            t3 = t0 + shiftar
            t4 = t1 + shiftar
            if (
                (b[tuple(t3)] == 0)
                and (b[tuple(t4)] == 0)
                and (np.min(t3) >= 1)
                and (np.min(t4) >= 1)
                and (t3[0] < boxSize[0] + 1)
                and (t3[1] < boxSize[1] + 1)
                and (t3[2] < boxSize[2] + 1)
                and (t4[0] < boxSize[0] + 1)
                and (t4[1] < boxSize[1] + 1)
                and (t4[2] < boxSize[2] + 1)
            ):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a
    return np.array(a) - 1