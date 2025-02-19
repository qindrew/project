#%%
import os
import matplotlib.pyplot as plt
import numpy as np
from ovito.io import import_file
from ase.io import read
from natsort import natsorted
from matplotlib.colors import LogNorm

#%%
def distance(atom1, atom2, cell_size=np.asarray([50,50,50])):
    diff = atom1-atom2
    diff = diff - np.round(diff / cell_size) * cell_size
    return np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)

dists = []
pipeline = import_file('snapshots.xyz')
nframes = len(pipeline.frames)
for i in range(nframes):
    positions = pipeline.compute(i).particles.positions.array
    atom1 = positions[14]
    atom2 = positions[17]
    dist = distance(atom1,atom2)
    dists.append(dist)

print(np.max(dists))
