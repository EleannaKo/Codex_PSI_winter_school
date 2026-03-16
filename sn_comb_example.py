import os
import numpy as np
import argparse
import multiprocessing as mp
from multiprocessing import Pool

# Ensure friendly behavior of OpenMP and multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
mp.set_start_method('spawn', force=True)

import pocomc as pc
from cosmoprimo import Cosmology
from scipy.stats import uniform, norm

# Import likelihood modules (new versions that load data from npz files)
from bao import BAOLikelihood
from cmb import CMBCompressedLikelihood
from supernova import Union3SNLikelihoodSys
from supernova_combined import SN_Combined
from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood
sn_data = np.load("SN_dataset.npy", allow_pickle=True).item()

sn_likelihoods = {}

for name, data in sn_data.items():
    _, z, mu, _, cov = data
    sn_likelihoods[name] = SN_Combined(z=z, mu=mu, cov=cov, name=name)

sn_like = sn_likelihoods['pantheonplus']

# for Om in [0.1, 0.3, 0.9]:
    # cosmo = Cosmology(Omega_m=Om, h=0.5, omega_b=0.022)
    # cosmo.set_engine('camb')
    # print(sn_like.calculate(cosmo))

cosmo = Cosmology(Omega_m=0.15815703, h=0.90029244, omega_b=0.022424)
cosmo.set_engine('camb')
print(sn_like.calculate(cosmo))

