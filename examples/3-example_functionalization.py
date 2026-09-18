#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
sys.path.append('..')
from src.nanocraft.cnfg import *
from src.nanocraft.log_cnfg import *
from src.nanocraft.utls import grid, run_parallel
from src.nanocraft.fnctn import functionalize

output_dir = "./results/functionalize"

settings = {
    'ligand_file': "./data/3-mpa-pp.xyz",
    'nanocrystal_file': glob.glob(os.path.join("./results/passivated", "*.xyz")),
    'density_type': "per_nm2",
    'density_value': [1.0, 2.0],
    'anchor_fragment': "SH",
    'placement_method': "distance",
    'random_angle': True,
    'random_seed': True
}

settings_grid = grid(settings)

from tqdm import tqdm
for settings in tqdm(settings_grid, desc="Progress"):
    functionalize(settings, output_dir=output_dir)
