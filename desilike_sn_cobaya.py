"""
Cobaya wrapper for desilike SN likelihoods

This allows PantheonPlusSNLikelihood, Union3SNLikelihood, and DESY5SNLikelihood
to be used as external likelihoods in Cobaya MCMC.

Usage in Cobaya:
    likelihood:
        sn:
            external: desilike_sn_cobaya.PantheonPlusSNWrapper
            # or Union3SNWrapper, or DESY5SNWrapper
"""

from cobaya.likelihood import Likelihood
import numpy as np


class PantheonPlusSNWrapper(Likelihood):
    """Wrapper for desilike PantheonPlusSNLikelihood"""
    
    def initialize(self):
        """Initialize the likelihood"""
        from desilike.likelihoods.supernovae import PantheonPlusSNLikelihood
        
        self.sn_like = PantheonPlusSNLikelihood()
        self.log.info(f"Initialized PantheonPlus with {len(self.sn_like.data)} SNe")
    
    def get_requirements(self):
        """Tell Cobaya we need cosmological parameters"""
        # We need distance modulus for each SN
        # This requires H0, Omega_m, w0_fld, wa_fld from the theory code
        return {}
    
    def logp(self, **params_values):
        """Calculate log-likelihood"""
        
        # Get Mb parameter
        Mb = params_values.get('Mb', -19.3)
        
        # Get cosmological parameters
        # These come from the theory code (CLASS/CAMB)
        cosmo = {
            'h': self.provider.get_param('h'),
            'Omega_m': self.provider.get_param('Omega_m'),
            'w0_fld': self.provider.get_param('w0_fld'),
            'wa_fld': self.provider.get_param('wa_fld'),
        }
        
        # Calculate chi-square
        try:
            # desilike likelihood expects these parameters
            chi2 = self.sn_like(Mb=Mb, **cosmo)
            return -0.5 * chi2
        except Exception as e:
            self.log.error(f"Likelihood calculation failed: {e}")
            return -np.inf


class Union3SNWrapper(Likelihood):
    """Wrapper for desilike Union3SNLikelihood"""
    
    def initialize(self):
        """Initialize the likelihood"""
        from desilike.likelihoods.supernovae import Union3SNLikelihood
        
        self.sn_like = Union3SNLikelihood()
        self.log.info(f"Initialized Union3")
    
    def get_requirements(self):
        """Tell Cobaya we need cosmological parameters"""
        return {}
    
    def logp(self, **params_values):
        """Calculate log-likelihood"""
        
        # Get Mb parameter
        dM = params_values.get('dM', -19.3)
        
        # Get cosmological parameters
        cosmo = {
            'h': self.provider.get_derived("h"),
            'Omega_m': self.provider.get_param('Omega_m'),
            'w0_fld': self.provider.get_param('w0_fld'),
            'wa_fld': self.provider.get_param('wa_fld'),
        }
        
        # Calculate chi-square
        try:
            chi2 = self.sn_like(dM=dM, **cosmo)
            return -0.5 * chi2
        except Exception as e:
            self.log.error(f"Likelihood calculation failed: {e}")
            return -np.inf


class DESY5SNWrapper(Likelihood):
    """Wrapper for desilike DESY5SNLikelihood (for DES-Dovekie)"""
    
    # Allow specifying custom data directory
    data_dir: str = None
    
    def initialize(self):
        """Initialize the likelihood"""
        from desilike.likelihoods.supernovae import DESY5SNLikelihood
        
        if self.data_dir is not None:
            self.sn_like = DESY5SNLikelihood(data_dir=self.data_dir)
            self.log.info(f"Initialized DES-Dovekie from {self.data_dir}")
        else:
            self.sn_like = DESY5SNLikelihood()
            self.log.info(f"Initialized DES-Y5 (default)")
        
        self.log.info(f"Number of SNe: {len(self.sn_like.data)}")
    
    def get_requirements(self):
        """Tell Cobaya we need cosmological parameters"""
        return {}
    
    def logp(self, **params_values):
        """Calculate log-likelihood"""
        
        # Get Mb parameter
        Mb = params_values.get('Mb', -19.3)
        
        # Get cosmological parameters
        cosmo = {
            'h': self.provider.get_param('h'),
            'Omega_m': self.provider.get_param('Omega_m'),
            'w0_fld': self.provider.get_param('w0_fld'),
            'wa_fld': self.provider.get_param('wa_fld'),
        }
        
        # Calculate chi-square
        try:
            chi2 = self.sn_like(Mb=Mb, **cosmo)
            return -0.5 * chi2
        except Exception as e:
            self.log.error(f"Likelihood calculation failed: {e}")
            return -np.inf