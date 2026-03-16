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
from desilike.likelihoods.supernovae import Union3SNLikelihood, PantheonPlusSNLikelihood, DESY5SNLikelihood
from compressed_sn import CompressedPantheonPlusSNLikelihood, CompressedUnion3SNLikelihood, CompressedDESY5SNLikelihood
from combined_sn import JointCompressedSNLikelihood

SN_NUISANCE_BOUNDS = {
    "pantheonplus": (-20.0, -18.0),   # dM
    "union3":       (-20.0, 20.0),   # dM 
    "desy5sn":      (-5, 5),   # Mb (DES Y5)
}

SN_NUISANCE_LABELS = {
    "pantheonplus": "dM_pantheon",
    "union3":       "dM_union3",
    "desy5sn":      "Mb_desy5",
}

SN_FILES = {
    "pantheonplus": "SN_data/pantheonplus_compressed.npz",
    "union3":       "SN_data/union3_compressed.npz",
    # "desy5sn":      "SN_data/desy5_compressed.npz",
    "desy5sn":      "SN_data/DES_Dovekey_compressed.npz",
}

_SN_INSTANCE = None
_SN_CLASS = None

def _get_sn_instance(sn_like_cls, cosmo, **init_kwargs):
    global _SN_INSTANCE, _SN_CLASS
    if sn_like_cls is None:
        return None

    if _SN_INSTANCE is None or (_SN_CLASS is not sn_like_cls):
        _SN_INSTANCE = sn_like_cls(cosmo=cosmo, **init_kwargs)
        _SN_CLASS = sn_like_cls
        return _SN_INSTANCE

    _SN_INSTANCE.cosmo = cosmo
    return _SN_INSTANCE


def build_bounds(model, include_bao, bao_sys, include_sn, sn_names, sn_sys_individual, sn_sys_correlation):
    """
    Build the list of parameter bounds.
    
    Order:
      - Cosmology:
         * For w0waCDM: [w0, wa, Omega_m, omega_b, h]
         * For LCDM: [Omega_m, omega_b, h] (w0=-1, wa=0 fixed)
      - If BAO is included and bao_sys True: add one BAO nuisance parameter.
      - If SN is included: add one SN nuisance parameter and/or one nuisance parameter for systematics.
    """
    bounds = []
    if model == 'w0waCDM':
        bounds += [[-3, 1], [-3, 2], [0.01, 0.99], [0.005, 0.1], [0.2, 1]]

    elif model == 'LCDM':
        bounds += [[0.01, 0.99], [0.005, 0.1], [0.2, 1]]
    else:
        raise ValueError("Unknown model option. Use 'w0waCDM' or 'LCDM'.")
    
    if include_bao and bao_sys:
        bounds += [[1, 9]]  # BAO nuisance parameter bound (systematics version)
    
    if include_sn:
        if "pantheonplus" in sn_names:
            bounds.append(list(SN_NUISANCE_BOUNDS["pantheonplus"]))
            if sn_sys_individual:
                bounds +=[[1, 5]] # Pantheon nuisance parameter bound
        if "union3" in sn_names:
            bounds.append(list(SN_NUISANCE_BOUNDS["union3"]))
            if sn_sys_individual:
                bounds +=[[1, 5]] # Union3 nuisance parameter bound
        if "desy5sn" in sn_names:
            bounds.append(list(SN_NUISANCE_BOUNDS["desy5sn"]))
            if sn_sys_individual:
                bounds +=[[1, 5]] # DES nuisance parameter bound
        if sn_sys_correlation:
            bounds +=[[-1, 1]]  # SN nuisance parameter in off diagonal covariance terms (SN dataset correlation)
            if "desy5sn" in sn_names:
                bounds +=[[-1,1]]
                bounds +=[[-1,1]] 
    return bounds

def prepPrior(bounds, bbn=None, bbnidx=3):
    """
    Create a pocomc Prior object for MCMC sampling.
    
    Each parameter gets a prior distribution:
      - Uniform over the provided bounds by default.
      - Optional Gaussian prior for a specific parameter (e.g., from BBN constraints).
    
    Parameters:
    - bounds: list of [min, max] for each parameter
    - bbn: optional [mean, std] for a Gaussian prior on one parameter
    - bbnidx: index of the parameter to apply the Gaussian prior
    
    Returns:
    - pc.Prior object used by PoCoMC to constrain the parameter space before sampling.
    """
    dists = [
        norm(bbn[0], bbn[1]) if (idx == bbnidx and bbn is not None)
        else uniform(lower, upper - lower)
        for idx, (lower, upper) in enumerate(bounds)
    ]
    return pc.Prior(dists)

def total_log_likelihood(params, model, bao_like=None, bao_sys=False, cmb_like=None, include_sn=False, sn_pantheon=None, sn_union3=None, sn_desy5=None, sn_sys_individual=False, sn_sys_correlation=False):
    """
    Combined likelihood for the chosen probes.
    """

    idx = 0
    if model == 'w0waCDM':
        if len(params) < 5:
            return -np.inf
        w0, wa, Omega_m, omega_b, h = params[idx:idx+5]
        idx += 5
    elif model == 'LCDM':
        if len(params) < 3:
            return -np.inf
        Omega_m, omega_b, h = params[idx:idx+3]
        w0, wa = -1.0, 0.0
        idx += 3
    else:
        raise ValueError("Unknown cosmological model.")
    
    
    # Enforce physical priors
    if (w0 + wa) >= 0:
        return -np.inf
    if (Omega_m * h**2) <= omega_b+0.0006441915396177796: 
        return -np.inf
    

    # Create a new Cosmology instance with the sampled parameters.
    cosmo = Cosmology(w0_fld=w0, wa_fld=wa, Omega_m=Omega_m, omega_b=omega_b, h=h, mnu=0.06, nnu=3.044, tau_reio=0.0544)
    cosmo.set_engine("camb")

    total_ll = 0.0

    # Update CMB likelihood model with the new cosmo.
    if cmb_like is not None:
        cmb_like.model = cosmo
        ll_cmb = cmb_like.calculate()
        total_ll += ll_cmb
        if total_ll == -np.inf:
            return -np.inf
    
    # Update BAO likelihood model with the new cosmo.
    if bao_like is not None:
        bao_like.model.cosmo = cosmo
        if bao_sys:
            if idx >= len(params):
                print('BAO issue, idx >= len(params)')
                return -np.inf
            bao_nuis = params[idx]
            idx += 1
            ll_bao = bao_like.calculate(sys_coeff=bao_nuis)
        else:
            ll_bao = bao_like.calculate()
        total_ll += ll_bao
        if total_ll == -np.inf:
            print('BAO issue, bao_like=-inf')
            return -np.inf

    
    # SN likelihood: create an instance using the shared cosmo and calculate    
    if include_sn:
        pantheon_sn = None
        union3_sn = None
        desy5_sn = None
        if sn_pantheon is not None:
            dM_pantheon = params[idx]; idx += 1
            pantheon_sn = _get_sn_instance(CompressedPantheonPlusSNLikelihood, cosmo, compressed_fn=SN_FILES["pantheonplus"],)
            if sn_sys_individual:
                sys_coeff_pantheon = params[idx]; idx +=1
            else: sys_coeff_pantheon = None

            if not sn_sys_correlation:
                pantheon_sn.calculate(dM=dM_pantheon, sys_coeff=sys_coeff_pantheon)
                ll_pantheon = pantheon_sn.loglikelihood
            else:
                ll_pantheon = 0.0
        else:
            ll_pantheon = 0.0

        if sn_union3 is not None:
            dM_union3 = params[idx]; idx += 1
            union3_sn = _get_sn_instance(CompressedUnion3SNLikelihood, cosmo, compressed_fn=SN_FILES["union3"])
            if sn_sys_individual:
                sys_coeff_union3 = params[idx]; idx +=1
            else: sys_coeff_union3 = None

            if not sn_sys_correlation:
                union3_sn.calculate(dM=dM_union3, sys_coeff=sys_coeff_union3)
                ll_union3 = union3_sn.loglikelihood
            else:
                ll_union3 = 0.0
        else:
            ll_union3 = 0.0

        if sn_desy5 is not None:
            Mb_desy5 = params[idx]; idx += 1
            desy5_sn = _get_sn_instance(CompressedDESY5SNLikelihood, cosmo, compressed_fn=SN_FILES["desy5sn"])
            if sn_sys_individual:
                sys_coeff_desy5 = params[idx]; idx +=1
            else: sys_coeff_desy5 = None
            if not sn_sys_correlation: 
                desy5_sn.calculate(Mb=Mb_desy5, sys_coeff=sys_coeff_desy5)
                ll_desy5 = desy5_sn.loglikelihood
            else: 
                ll_desy5 = 0.0
        else:
            Mb_desy5 = 0.0
            sys_coeff_desy5 = 1.0
            ll_desy5 = 0.0

        if sn_sys_correlation:
            r_ab = params[idx]; idx +=1
            r_ad = params[idx]; idx +=1
            r_bd = params[idx]; idx +=1
            joint_sn = JointCompressedSNLikelihood(pantheon_like=pantheon_sn, union_like=union3_sn, des_like=desy5_sn, cosmo=None)
            ll_sn = joint_sn.calculate(dM1=dM_pantheon, dM2=dM_union3, Mb3=Mb_desy5, s1=sys_coeff_pantheon, s2=sys_coeff_union3, s3=sys_coeff_desy5, R_AB=r_ab, R_AD=r_ad, R_BD=r_bd, desyr5=sn_desy5)
        else:
            ll_sn = ll_pantheon + ll_union3 + ll_desy5 
        
        if not np.isfinite(ll_sn):
            return -np.inf

        total_ll += ll_sn

    if idx != len(params):
        return -np.inf

    return total_ll

def main(args):
    
    probe_list = [p.strip().upper() for p in args.likelihoods.split(',')]
    include_bao = 'BAO' in probe_list
    include_cmb = 'CMB' in probe_list
    include_sn  = 'SN'  in probe_list
    sn_names = [s.strip().lower() for s in args.sn_likelihood.split(",")]
    print('probe list: ',probe_list)
    model = args.model  # 'w0waCDM' or 'LCDM'
    
    # init BAO likelihood.
    bao_like = None
    if include_bao:
        dataset = args.bao_dataset  
        bao_like = BAOLikelihood(data_dir=args.data_dir, dataset=dataset, engine='camb')
    
    # init CMB likelihood.
    cmb_like = None
    if include_cmb:
        cmb_like = CMBCompressedLikelihood(engine='camb', cosmo=None)

    # init SN likelihoods.
    sn_pantheon = None
    sn_union3   = None
    sn_desy5    = None

    if include_sn:
        if "pantheonplus" in sn_names:
            sn_pantheon = CompressedPantheonPlusSNLikelihood(
                cosmo=None,
                compressed_fn="SN_data/pantheonplus_compressed.npz"
            )

        if "union3" in sn_names:
            sn_union3 = CompressedUnion3SNLikelihood(
                cosmo=None,
                compressed_fn="SN_data/union3_compressed.npz"
            )

        if "desy5sn" in sn_names:
            sn_desy5 = CompressedDESY5SNLikelihood(
                cosmo=None,
                compressed_fn="SN_data/DES_Dovekey_compressed.npz"
                # compressed_fn="SN_data/desy5_compressed.npz"
            )
    
    # Build priors
    bounds = build_bounds(model, include_bao, args.bao_sys, include_sn, sn_names, args.sn_sys_individual, args.sn_sys_correlation)
    if include_cmb:
        prior = prepPrior(bounds)
    else:
        if model =='LCDM':
            prior = prepPrior(bounds,[0.02218,0.00055],bbnidx=1)  
        else:
            prior = prepPrior(bounds,[0.02218,0.00055])    


    print("Total parameters:", len(bounds))
    print("Cosmo params:", 5 if model == "w0waCDM" else 3)
    print("BAO params:", int(include_bao and args.bao_sys))
    print("Number of SN datasets:", len(sn_names))
    print("Number of SN nuisance parameters:", len(sn_names))

    
    # Use multiprocessing pool.
    with Pool(args.ncores) as pool:
        sampler = pc.Sampler(
            prior=prior,
            likelihood=total_log_likelihood,
            vectorize=False,
            pool=pool,
            output_dir=args.output_dir,
            output_label=args.output_label,
            likelihood_kwargs={'model': model,
                            'bao_like': bao_like,
                            'bao_sys': args.bao_sys,
                            'cmb_like': cmb_like,
                            'include_sn': include_sn,
                            'sn_pantheon': sn_pantheon,
                            'sn_union3': sn_union3,
                            'sn_desy5': sn_desy5,
                            'sn_sys_individual': args.sn_sys_individual,
                            'sn_sys_correlation': args.sn_sys_correlation
                            }
            )
        sampler.run(n_total=8192)
    # sampler.run(n_total=1000)
    
    samples, weights, logl, logp = sampler.posterior()
    output_file = os.path.join(args.output_dir, args.output_label + '.txt')
    np.savetxt(output_file, np.column_stack((samples, weights, logl, logp)),
               header='samples weight logl logp')
    print(f"Sampling complete. Results saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run joint likelihood sampler for BAO, CMB, and SN.")
    parser.add_argument("--data_dir", type=str, default="bao_data",
                        help="Directory containing data files (npz files for BAO, etc.)")
    parser.add_argument("--sn_data_dir", type=str, default="sn_compressed",
                        help="Directory containing compressed SN .npz files")
    parser.add_argument("--likelihoods", type=str, default="BAO,CMB,SN",
                        help="Comma-separated list of probes to include (choose from BAO, CMB, SN)")
    parser.add_argument("--bao_dataset", type=str, default="DESI_SDSS",
                        help="BAO dataset to use: 'DESI_SDSS' or 'DESI'")
    parser.add_argument("--bao_sys", action="store_true",
                        help="Use BAO likelihood with systematics (adds one extra parameter)")
    parser.add_argument("--sn_likelihood", type=str, default="union3, pantheonplus, desy5sn",
                        help="SN likelihood to use: union3, pantheonplus, or desy5sn")
    parser.add_argument("--sn_sys_individual", action="store_true",
                    help="Use SN likelihood with systematics per dataset (adds one extra parameterper SN dataset)")
    parser.add_argument("--sn_sys_correlation", action="store_true",
                    help="Combine SN datasets accounting for correlation between them (adds one extra parameter)")
    parser.add_argument("--model", type=str, default="w0waCDM",
                        help="Cosmological model: 'w0waCDM' (sample w0,wa) or 'LCDM' (fix w0=-1, wa=0)")
    parser.add_argument("--output_dir", type=str, default="chains/",
                        help="Directory to save the chain files")
    parser.add_argument("--output_label", type=str, default="chain_joint",
                        help="Label for the output chain file")
    parser.add_argument("--ncores", type=int, default=16,
                        help="Number of cores to use for parallel processing")
    args = parser.parse_args()
    main(args)