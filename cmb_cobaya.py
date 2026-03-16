import numpy as np
from cobaya.model import get_model
from cobaya.log import LoggedError

class PlanckCobayaLikelihood:
    _model = None   # shared Cobaya model across all instances

    @classmethod
    def load_model(cls):
        """
        Loads the Cobaya model (Planck 2018 TTTEEE + lensing + CAMB).
        Only runs once for speed.
        """
        if cls._model is not None:
            return

        info = {
            "theory": {
                "camb": {
                    "accuracy_level": 1,
                    "lens_potential_accuracy": 1
                }
            },
            "params": {
                # Core cosmological parameters (free)
                "ombh2": None,
                "omch2": None,
                "H0": None,
                "tau": None,
                "ns": None,
                "As": None,
                # Optional: dark energy extensions
                "w": None,
                "wa": None,

                # --- All Planck nuisance parameters free ---
                "A_cib_217": None,
                "A_planck": None,
                "A_sz": None,
                "calib_100T": None,
                "calib_217T": None,
                "gal545_A_100": None,
                "gal545_A_143": None,
                "gal545_A_143_217": None,
                "gal545_A_217": None,
                "galf_TE_A_100": None,
                "galf_TE_A_100_143": None,
                "galf_TE_A_100_217": None,
                "galf_TE_A_143": None,
                "galf_TE_A_143_217": None,
                "galf_TE_A_217": None,
                "ksz_norm": None,
                "ps_A_100_100": None,
                "ps_A_143_143": None,
                "ps_A_143_217": None,
                "ps_A_217_217": None,
                "xi_sz_cib": None,
            },

            "likelihood": {
                "planck_2018_highl_plik.TTTEEE": None,
                "planck_2018_lowl.TT": None,
                "planck_2018_lowl.EE": None,
                "planck_2018_lensing.clik": None,
            }
        }

        cls._model = get_model(info)

    def __init__(self):
        PlanckCobayaLikelihood.load_model()
        self.model = PlanckCobayaLikelihood._model