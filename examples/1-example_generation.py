#!/usr/bin/env python3
import sys
import numpy as np
sys.path.append('..')
from src.nanocraft.cnfg import *
from src.nanocraft.log_cnfg import *
from src.nanocraft.utls import grid, run_parallel
from src.nanocraft.sprcll import load_supercell
from src.nanocraft.ncgen import generate

output_dir = "./results"

supercell = load_supercell(
    inputfile="./data/1540817.cif",
    supercell_size=[10, 10, 10]
)

settings = {
 'rcut': np.arange(.5, 1.21, 0.01),
 'fratio': 1.0,
 'mask_shape': ["sphere"],
 'central_site': [['Cd', 'Te']],
 'acceptance_criterion': 'auto'
}

settings_grid = grid(settings, collapse=('mask_shape', 'fratio', ['sphere', 'cube', 'octahedron', 'tetrahedron']))


## for parallel execution uncomment this and comment whats bellow
#run_parallel(generate, settings_grid, 8, supercell=supercell, output_dir=output_dir)

from tqdm import tqdm
for settings in tqdm(settings_grid, desc="Progress"):
    generate(settings=settings, supercell=supercell, output_dir=output_dir)
