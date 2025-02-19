#!/usr/bin/env python3

"""
Example SpectralABF simulation with pysages and lammps.

For a list of possible options for running the script pass `-h` as argument from the
command line, or call `get_args(["-h"])` if the module was loaded interactively.
"""

# %%
import argparse
import sys
import os
import importlib
import numpy as onp
from lammps import lammps

import pysages
import jax
from jax import numpy as np
from pysages.colvars.core import CollectiveVariable, multicomponent
from pysages.methods import CVRestraints
from FSpectralABF import FSpectralABF, Funnel_Logger
import dill as pickle
from funnel_function import get_funnel_force
#%%
class Mean_Dist(CollectiveVariable):
    def __init__(self, indices, box=np.asarray([18,18,18])):
        super().__init__(indices)
        self.even_idx = np.arange(len(indices))[::2]
        self.odd_idx = np.arange(len(indices))[1::2]
        self.box = box
    @property
    def function(self):
        return lambda r: mean_dist(r, self.even_idx, self.odd_idx, box=self.box)

def mean_dist(r, even_idx, odd_idx, box=np.asarray([18, 18, 18])):
    diffs = r[even_idx] - r[odd_idx]
    periodic_diffs = diffs - np.round(diffs / box) * box
    distances = np.sqrt(np.sum(periodic_diffs**2, axis=1) + 1e-12)
    mean = np.mean(distances)
    return mean

def generate_context(args="", script="run.in", store_freq=1):
    """
    Returns a lammps simulation defined by the contents of `script` using `args` as
    initialization arguments.
    """
    context = lammps(cmdargs=args.split())
    context.file(script)
    # Allow for the retrieval of the wrapped positions
    context.command(f"dump 5 all custom {store_freq} dump.coords id type x y z ix iy iz vx vy vz")
    context.command("dump_modify 5 sort id")
    #context.command(f"fix unwrap all store/state {store_freq} xu yu zu")
    return context


def get_args(argv):
    available_args = [
        ("time-steps", "t", int, 1e7, "Number of simulation steps"),
        ("kokkos", "k", bool, True, "Whether to use Kokkos acceleration"),
        ("log-steps", "l", int, 10000, "Number of simulation steps for logging"),
        ("lower", "lo", tuple, None, "Lower corner Grid Boundary"),
        ("upper", "up", tuple, None, "Upper corner Grid Boundary"),
        ("cvend", "e", float, 0, "ending CV value of steered MD")
    ]
    
    parser = argparse.ArgumentParser(description="Example script to run pysages with lammps")

    for name, short, T, val, doc in available_args:
        if T is bool:
            parser.add_argument(f"--{name}", f"-{short}", action="store_true" if val else "store_false", help=doc)
        elif T is tuple:
            # Lambda function to strip parentheses and split the string into a tuple of floats
            parser.add_argument(f"--{name}", f"-{short}", type=lambda x: tuple(map(float, x.strip("()").split(','))), default=val, help=doc)
        else:
            convert = (lambda x: int(float(x))) if T is int else T
            parser.add_argument(f"--{name}", f"-{short}", type=convert, default=T(val), help=doc)

    parser.add_argument("--filepath", "-f", type=str, default=None, help="Filepath for the input data")
    
    return parser.parse_args(argv)


def get_executor():
    futures = importlib.import_module("mpi4py.futures")
    return futures.MPIPoolExecutor(max_workers=1)

def main(argv):

    args = get_args(argv)

    context_args = {"store_freq": args.log_steps}
    context_args["args"] = "-k on g 1 -sf kk -pk kokkos neigh half newton on"
    with open('restart.pkl', "rb") as f:
        state = pickle.load(f)
    state = pysages.run(state, generate_context, args.time_steps, context_args=context_args, executor=get_executor())
    with open('restart2.pkl', "wb") as f:
        pickle.dump(raw_result, f)
if __name__ == "__main__":
    main(sys.argv[1:])
