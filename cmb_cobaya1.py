from cobaya.model import get_model
import numpy as np
from scipy.stats import uniform

# Define your model and data
info = {
    "theory": {"camb": None},
    "params": {
        # cosmological parameters
        "ombh2": {"prior": {"min": 0.01, "max": 0.04}, "ref": 0.02218},
        "omch2": {"prior": {"min": 0.05, "max": 0.25}, "ref": 0.1203},
        "H0": {"prior": {"min": 40, "max": 100}, "ref": 67.36}, # check if its zeta instead of H0
        "tau": {"prior": {"min": 0.01, "max": 0.8}, "ref": 0.054},
        "ns": {"prior": {"min": 0.9, "max": 1.1}, "ref": 0.965},
        "As": {"prior": {"min": 1e-10, "max": 5e-9}, "ref": 2.1e-9, "latex": "A_s"},
        "w": {"prior": {"min": -2, "max": 0}, "ref": -1},
        "wa": {"prior": {"min": -2, "max": 2}, "ref": 0},

        # nuisance parameters (Planck 2018 high-l TTTEEE) I used cobaya-doc planck_2018_highl_plik.TTTEEE --python
        "A_cib_217": {"prior": {"dist": "uniform", "min": 0, "max": 200}, "ref": 67, "latex": "A^\\mathrm{CIB}_{217}"},
        "A_planck": {"prior": {"dist": "norm", "loc": 1, "scale": 0.0025}, "ref": 1, "latex": "y_\\mathrm{cal}"},
        "A_sz": {"prior": {"dist": "uniform", "min": 0, "max": 10}, "ref": 7, "latex": "A^\\mathrm{tSZ}_{143}"},
        "calib_100T": {"prior": {"dist": "norm", "loc": 1.0002, "scale": 0.0007}, "ref": 1.0002, "latex": "c_{100}"},
        "calib_217T": {"prior": {"dist": "norm", "loc": 0.99805, "scale": 0.00065}, "ref": 0.99805, "latex": "c_{217}"},

        "gal545_A_100": {"prior": {"dist": "norm", "loc": 8.6, "scale": 2}, "ref": 8.6, "latex": "A^\\mathrm{dustTT}_{100}"},
        "gal545_A_143": {"prior": {"dist": "norm", "loc": 10.6, "scale": 2}, "ref": 10.6, "latex": "A^\\mathrm{dustTT}_{143}"},
        "gal545_A_143_217": {"prior": {"dist": "norm", "loc": 23.5, "scale": 8.5}, "ref": 23.5, "latex": "A^\\mathrm{dustTT}_{143\\times217}"},
        "gal545_A_217": {"prior": {"dist": "norm", "loc": 91.9, "scale": 20}, "ref": 91.9, "latex": "A^\\mathrm{dustTT}_{217}"},

        # fixed EE dust parameters
        "galf_EE_A_100": {"value": 0.055, "ref": 0.055, "latex": "A^\\mathrm{dustEE}_{100}"},
        "galf_EE_A_100_143": {"value": 0.04, "ref": 0.04, "latex": "A^\\mathrm{dustEE}_{100\\times143}"},
        "galf_EE_A_100_217": {"value": 0.094, "ref": 0.094, "latex": "A^\\mathrm{dustEE}_{100\\times217}"},
        "galf_EE_A_143": {"value": 0.086, "ref": 0.086, "latex": "A^\\mathrm{dustEE}_{143}"},
        "galf_EE_A_143_217": {"value": 0.21, "ref": 0.21, "latex": "A^\\mathrm{dustEE}_{143\\times217}"},
        "galf_EE_A_217": {"value": 0.7, "ref": 0.7, "latex": "A^\\mathrm{dustEE}_{217}"},

        # TE dust parameters
        "galf_TE_A_100": {"prior": {"dist": "norm", "loc": 0.13, "scale": 0.042}, "ref": 0.13, "latex": "A^\\mathrm{dustTE}_{100}"},
        "galf_TE_A_100_143": {"prior": {"dist": "norm", "loc": 0.13, "scale": 0.036}, "ref": 0.13, "latex": "A^\\mathrm{dustTE}_{100\\times143}"},
        "galf_TE_A_100_217": {"prior": {"dist": "norm", "loc": 0.46, "scale": 0.09}, "ref": 0.46, "latex": "A^\\mathrm{dustTE}_{100\\times217}"},
        "galf_TE_A_143": {"prior": {"dist": "norm", "loc": 0.207, "scale": 0.072}, "ref": 0.207, "latex": "A^\\mathrm{dustTE}_{143}"},
        "galf_TE_A_143_217": {"prior": {"dist": "norm", "loc": 0.69, "scale": 0.09}, "ref": 0.69, "latex": "A^\\mathrm{dustTE}_{143\\times217}"},
        "galf_TE_A_217": {"prior": {"dist": "norm", "loc": 1.938, "scale": 0.54}, "ref": 1.938, "latex": "A^\\mathrm{dustTE}_{217}"},

        # kSZ and point sources
        "ksz_norm": {"prior": {"dist": "uniform", "min": 0, "max": 10}, "ref": 0, "latex": "A^\\mathrm{kSZ}"},
        "ps_A_100_100": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 257, "latex": "A^\\mathrm{PS}_{100}"},
        "ps_A_143_143": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 47, "latex": "A^\\mathrm{PS}_{143}"},
        "ps_A_143_217": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 40, "latex": "A^\\mathrm{PS}_{143\\times217}"},
        "ps_A_217_217": {"prior": {"dist": "uniform", "min": 0, "max": 400}, "ref": 104, "latex": "A^\\mathrm{PS}_{217}"},
        "xi_sz_cib": {"prior": {"dist": "uniform", "min": 0, "max": 1}, "ref": 0, "latex": "\\xi^{\\mathrm{tSZ}\\times\\mathrm{CIB}}"},

    },
    "likelihood": {
        "planck_2018_highl_plik.TTTEEE": None,
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_lensing.clik": None,
    },
    # sampler can be None if I just want likelihoods
    "sampler": None,
    "output": None  # no chains needed
}

# Build the model
model = get_model(info)

# Extract names of parameters that have priors (i.e. are sampled)
sampled_param_names = [
    name for name, par in info["params"].items()
    if "prior" in par
]

# Reference point vector
ref_params = np.array([info["params"][n]["ref"] for n in sampled_param_names])


def cobaya_loglike_only(params_array):
    point = {name: float(val) for name, val in zip(sampled_param_names, params_array)}
    try:
        lp_obj = model.logposterior(point)  
        # Cobaya returns a LogPosterior object:
        # lp_obj.logpost, lp_obj.logprior, lp_obj.loglike

        loglike = float(lp_obj.loglike)

        if not np.isfinite(loglike):
            return -np.inf
        return loglike

    except Exception as e:
        print("Cobaya eval failed for point:", point)
        print("Exception:", repr(e))
        return -np.inf


# Test
test = np.array([info["params"][n]["ref"] for n in sampled_param_names])
print("Likelihood at reference params =", cobaya_loglike_only(test))
