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
import pickle
import importlib
import numpy as onp
from lammps import lammps

import pysages
import jax
from pysages.colvars import Distance
from pysages.methods import HistogramLogger
from jax import numpy as np
from pysages.colvars.core import CollectiveVariable, multicomponent
from pysages.methods import Metadynamics, MetaDLogger, CVRestraints, SpectralABF, ABF
from pysages.methods import Steered, SteeredLogger
from jax import grad, jacrev, jit
from pysages.methods import Unbiased
from pysages.backends import SamplingContext
from ase.io import read
#%%
def generate_context(args="", script="run.in", store_freq=1000):
    """
    Returns a lammps simulation defined by the contents of `script` using `args` as
    initialization arguments.
    """
    context = lammps(cmdargs=args.split())
    context.file(script)
    # Allow for the retrieval of the wrapped positions
    context.command(f"dump 5 all custom {store_freq} asdf.coords id type xu yu zu")
    context.command("dump_modify 5 sort id")
    return context

def get_args(argv):
    available_args = [
        ("time-steps", "t", int, 1e4, "Number of simulation steps"),
        ("kokkos", "k", bool, True, "Whether to use Kokkos acceleration"),
        ("log-steps", "l", int, 1000, "Number of simulation steps for logging"),
        ("lower", "lo", tuple, None, "Lower corner Grid Boundary"),
        ("upper", "up", tuple, None, "Upper corner Grid Boundary"),
        ("cvend", "e", tuple, 0, "ending CV value of steered MD"),
        ("subset", "s", list, None, "select subset of atoms")
    ]
    
    parser = argparse.ArgumentParser(description="Example script to run pysages with lammps")

    for name, short, T, val, doc in available_args:
        if T is bool:
            parser.add_argument(f"--{name}", f"-{short}", action="store_true" if val else "store_false", help=doc)
        elif T is tuple:
            # Lambda function to strip parentheses and split the string into a tuple of floats
            parser.add_argument(f"--{name}", f"-{short}", type=lambda x: tuple(map(float, x.strip("()").split(','))), default=val, help=doc)
        elif T is list:
            parser.add_argument(f"--{name}", f"-{short}", type=lambda x: list(map(int, x.strip("[]").split(','))), default=val, help=doc)
        else:
            convert = (lambda x: int(float(x))) if T is int else T
            parser.add_argument(f"--{name}", f"-{short}", type=convert, default=T(val), help=doc)

    parser.add_argument("--filepath", "-f", type=str, default=None, help="Filepath for the input data")
    parser.add_argument("--r0", "-r", type=float, default=None, help="Filepath for the input data")   
    parser.add_argument("--nexp", "-n", type=float, default=None, help="Filepath for the input data")  
    return parser.parse_args(argv)


def get_executor():
    futures = importlib.import_module("mpi4py.futures")
    return futures.MPIPoolExecutor(max_workers=1)

#%%
def steer(argv):
    args = get_args(argv)
    context_args = {"store_freq": args.log_steps}
    context_args["args"] = "-k on g 1 -sf kk -pk kokkos neigh half newton on"

    cvs = [Distance([14, 17]),Distance([52, 55]),Distance([90, 93]),Distance([128, 131]),Distance([166, 169]),Distance([204, 207]),Distance([242, 245]),Distance([280, 283]),]

    starting_CV = np.asarray([1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5])
    time_steps = 3.001e7#args.time_steps
    ending_CV = np.asarray([5,5,5,5,5,5,5,5])
    velocity = (ending_CV-starting_CV) / time_steps / 0.2
    callback = SteeredLogger('log.steered', args.log_steps)
    method = Steered(cvs, np.asarray([4,4,4,4,4,4,4,4]), starting_CV, velocity)
    raw_result = pysages.run(
       method,
       generate_context,
       args.time_steps,
       callback=callback,
       context_args=context_args,
       executor=get_executor())
    pysages.save(raw_result, "restart2.pkl")
if __name__ == "__main__":
    steer(sys.argv[1:])
