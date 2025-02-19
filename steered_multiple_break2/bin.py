#%%
import argparse
import sys
import os
import pickle
import importlib
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file, export_file
from natsort import natsorted
import glob
from tqdm import tqdm
from ase.io import read, write

def compute_distances(atom1, atom2, cell_size=np.asarray([18,18,18])):
    diff = atom1-atom2
    diff = diff - np.round(diff / cell_size) * cell_size
    return np.sqrt(np.sum(diff**2) + 1e-12)

pipeline = import_file('combined_dump.coords')
nframes = len(pipeline.frames)
dists = np.zeros(nframes)
for i in tqdm(range(nframes)):
    r = pipeline.compute(i).particles.positions.array
    dists[i] = compute_distances(r[17],r[14])

hist, bin_edges = np.histogram(dists, bins=50)

# Get bin index for each distance
bin_indices = np.digitize(dists, bin_edges)

def sample_bin(indices):
    if len(indices) >= 20:
        return np.random.choice(indices, 20, replace=True)
    else:
        return np.array([])

pipeline = import_file('combined_dump.coords')
counter = 0
for i in range(len(bin_edges)):
    if i == 0:
        print(f"Bin 0 (values below {bin_edges[0]:.3f}): not sampled")
        continue
    indices = np.where(bin_indices == i)[0]
    sampled = sample_bin(indices)
    if len(sampled) > 1:
        for j in sampled:
            counter += 1
            export_file(pipeline, f"output/output.{counter}.data", "lammps/data", frame=j)
        print(f"Bin {i} (range: {bin_edges[i-1]:.3f} - {bin_edges[i]:.3f}): 20 points sampled")
    else:
        print(f"Bin {i} (range: {bin_edges[i-1]:.3f} - {bin_edges[i]:.3f}): not sampled")

for i in natsorted(os.listdir('output')):
    atoms = read('output/'+i,format='lammps-data',atom_style='atomic',Z_of_type={1:6, 2:1},read_image_flags=False)
    del atoms.arrays['type']
    del atoms.arrays['id']
    del atoms.arrays['momenta']
    write('snapshots.xyz',atoms,format='extxyz',append=True)


pipeline = import_file('snapshots.xyz')
nframes = len(pipeline.frames)
dists = np.zeros(nframes)
for i in tqdm(range(nframes)):
    r = pipeline.compute(i).particles.positions.array
    dists[i] = compute_distances(r[17],r[14])

plt.hist(dists, bins=bin_edges, alpha=0.1, color='r')
plt.xlabel('bond distance')
plt.ylabel('count')
plt.show()

