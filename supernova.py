import numpy as np
from desilike.likelihoods.supernovae import Union3SNLikelihood
from numpy.linalg import inv

class Union3SNLikelihoodSys:

    def __init__(self, cosmo=None, data_dir=None):
        self.sn = Union3SNLikelihood(data_dir=data_dir)
        self.cosmo = cosmo
        self.loglikelihood = None

    def calculate(self, dM=None, sys_coeff=None):
        union3 = Union3SNLikelihood(cosmo = self.cosmo)
        union3.calculate(dM = dM)
        cov = union3.covariance
        cov_sys = sys_coeff * cov

        logL = union3.loglikelihood
        chi2 = -2 * logL

        chi2_sys = chi2/sys_coeff
        self.loglikelihood = -0.5 * (chi2_sys + np.log(np.linalg.det(cov_sys)))

        return float(self.loglikelihood)
