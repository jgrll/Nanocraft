#!/usr/bin/env python3
import os
import sys
import numpy as np
sys.path.append('..')
from src.nanocraft.cnfg import *
from src.nanocraft.log_cnfg import *
from src.nanocraft.utls import grid, run_parallel
from src.nanocraft.pssv import passivate

output_dir = "./results/passivated"

settings = {
    'nanocrystal_file': glob.glob(os.path.join("./results/sphere_CdTe", "*.xyz")),
    'passiv_scheme': [[('Cd', 'SH'), ('Te', 'H')]],
    'binding_site':  ['ontop'],
    'facet_constrain': ['all'],
    'coord_constrain': ['all'],
    'neutral_reach': 'pseudorandom',
    'oxidation_method': 'all_ox'
}

settings_grid = grid(settings)

from tqdm import tqdm
for settings in tqdm(settings_grid, desc="Progress"):
    passivate(settings, output_dir=output_dir)
