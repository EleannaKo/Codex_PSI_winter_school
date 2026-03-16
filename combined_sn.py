import numpy as np
from desilike.likelihoods.base import BaseLikelihood, BaseGaussianLikelihood 
from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood
from compressed_sn import CompressedPantheonPlusSNLikelihood, CompressedUnion3SNLikelihood, CompressedDESY5SNLikelihood
from scipy.linalg import block_diag, sqrtm


class JointCompressedSNLikelihood:

    name = "JointCompressedSN"

    def __init__(self, pantheon_like=None, union_like=None, des_like=None, cosmo=None):
        self.pantheon = pantheon_like
        self.union = union_like
        self.des = des_like
        self.cosmo = cosmo

        # dimensions
        self.n1 = self.pantheon.covariance.shape[0]
        self.n2 = self.union.covariance.shape[0]
        if des_like is not None:
            self.n3 = self.des.covariance.shape[0]

    
    def calculate(self, dM1=0.0, dM2=0.0, Mb3=0.0, s1=1.0, s2=1.0, s3=1.0, r=0.0, R_AB=0.0, R_AD=0.0, R_BD=0.0, desyr5=None):
        if s1 is None:
            s1 = 1.0
        if s2 is None:
            s2 = 1.0
        if s3 is None:
            s3 = 1.0

        def combine_covariances(A, B, R):
            sqrtA = sqrtm(A)
            sqrtB = sqrtm(B)
            sqrtA = np.real(sqrtA)
            sqrtB = np.real(sqrtB)
            C = R * np.dot(sqrtA, sqrtB)
            M = np.block([[A, C],
                        [C.T, B]])
            return M 
        
        def combine_three_covariances(A, B, D, R_AB, R_AD, R_BD):
            # Matrix square roots
            sqrtA = np.real(sqrtm(A))
            sqrtB = np.real(sqrtm(B))
            sqrtD = np.real(sqrtm(D))

            # Cross-covariances
            C_AB = R_AB * np.dot(sqrtA, sqrtB)
            C_AD = R_AD * np.dot(sqrtA, sqrtD)
            C_BD = R_BD * np.dot(sqrtB, sqrtD)

            # Block matrix
            M = np.block([
                [A,        C_AB,       C_AD],
                [C_AB.T,   B,          C_BD],
                [C_AD.T,   C_BD.T,     D]
            ])

            return M
    
        # ---- Compute theory + data for each dataset ----
        self.pantheon.calculate(dM=dM1, sys_coeff=None)
        self.union.calculate(dM=dM2, sys_coeff=None)
        delta1 = self.pantheon.flatdata - self.pantheon.flattheory
        delta2 = self.union.flatdata - self.union.flattheory
        

        # ---- Build covariance blocks ----
        C1 = self.pantheon.covariance
        C2 = self.union.covariance            
    
        C11 = s1 * C1
        C22 = s2 * C2

        if desyr5 is not None:
            self.des.calculate(Mb=Mb3, sys_coeff=None)
            delta3 = self.des.flatdata - self.des.flattheory
            C3 = self.des.covariance
            C33 = s3 * C3
            C_full = combine_three_covariances(C11, C22, C33, R_AB, R_AD, R_BD)
            delta = np.concatenate([delta1, delta2, delta3])
        else: 
            C_full = combine_covariances(C11, C22, r)
            delta = np.concatenate([delta1, delta2])


        # ---- Likelihood ----
        sign, logdet = np.linalg.slogdet(C_full)
        if sign <= 0:
            return -np.inf  # not positive definite

        Cinv = np.linalg.inv(C_full)
        chi2 = delta @ Cinv @ delta

        self.loglikelihood = -0.5 * (chi2 + logdet)

        return float(self.loglikelihood)
