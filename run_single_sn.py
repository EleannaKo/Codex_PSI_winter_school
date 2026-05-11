#!/usr/bin/env python
"""
单个 SN 数据集分析（使用官方 desilike likelihood）

可选择的数据集：
- PantheonPlus (PantheonPlusSNLikelihood)
- Union3 (Union3SNLikelihood)
- DES-Dovekie (DESY5SNLikelihood)

每个数据集使用完整的原始数据，不是压缩的
"""

import sys
from pathlib import Path
# sys.path.insert(0, str(Path.home() / "PSIWinterSchool2025"))

from cobaya.run import run

# ============================================================================
# Configuration
# ============================================================================
OMEGA_NU_FIXED = 0.0006441915396177796
# 选择数据集（修改这里！）
DATASET = "union3"  # 选项: "pantheonplus", "union3", "dovekie"

# ============================================================================
# Parameters
# ============================================================================

def get_params():
    """Cosmological parameters (w0wa CDM)"""
    
    params = {
        # ====================================================================
        # Cosmological parameters
        # ====================================================================
        
        "omega_b": {
            "prior": {"dist": "norm", "loc": 0.02218, "scale": 0.00055},
            "ref": {"dist": "norm", "loc": 0.02218, "scale": 0.00055},
            "proposal": 0.0001,
            "latex": r"\omega_b",
        },
        
        "Omega_m": {
            "prior": {"min": 0.01, "max": 0.99},
            "ref": {"dist": "norm", "loc": 0.30, "scale": 0.05},
            "proposal": 0.01,
            "latex": r"\Omega_m",
            "drop": True,
        },
        
        "H0": {
            "prior": {"min": 55, "max": 85},
            "ref": {"dist": "norm", "loc": 70, "scale": 3},
            "proposal": 0.5,
            "latex": "H_0",
        },
        
        # w0wa CDM dark energy
        "w0_fld": {
            "prior": {"min": -2.0, "max": -0.3},
            "ref": {"dist": "norm", "loc": -1.0, "scale": 0.1},
            "proposal": 0.05,
            "latex": r"w_0",
        },
        
        "wa_fld": {
            "prior": {"min": -2, "max": 2},
            "ref": {"dist": "norm", "loc": 0.0, "scale": 0.3},
            "proposal": 0.1,
            "latex": r"w_a",
        },
        
        # ====================================================================
        # Derived parameters
        # ====================================================================
        
        "h": {
            "derived": "lambda H0: H0/100",
            "latex": r"h",
        },
        
        "omega_cdm": {
            "value": f"lambda Omega_m, omega_b, H0: Omega_m*(H0/100.)**2 - omega_b - {OMEGA_NU_FIXED}",
            "latex": r"\omega_{\rm cdm}",
        },
    }
    
    return params


def get_sn_params():
    """SN nuisance parameters"""
    
    params = {
        "dM": {
            "prior": {"min": -25, "max": -15},
            "ref": {"dist": "norm", "loc": -19.3, "scale": 0.5},
            "proposal": 0.1,
            "latex": r"M_b",
        },
    }
    
    return params


# ============================================================================
# Likelihoods
# ============================================================================

def get_sn_likelihood(dataset):
    """Get SN likelihood based on dataset choice - using wrapper"""
    
    # Import the wrapper classes
    import sys
    from pathlib import Path
    # sys.path.insert(0, str(Path.home() / "PSIWinterSchool2025"))
    
    from desilike_sn_cobaya import (
        PantheonPlusSNWrapper,
        Union3SNWrapper,
        DESY5SNWrapper
    )
    
    if dataset.lower() == "pantheonplus":
        return {
            "sn": {
                "external": PantheonPlusSNWrapper,
                "params": {"Mb": None},
            }
        }
    
    elif dataset.lower() == "union3":
        return {
            "sn": {
                "external": Union3SNWrapper,
                "params": {"dM": None},
            }
        }
    
    elif dataset.lower() == "dovekie":
        return {
            "sn": {
                "external": DESY5SNWrapper,
                "data_dir": "/home/h774li/PSIWinterSchool2025/sn_data/DES-Dovekie",
                "params": {"Mb": None},
            }
        }
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose: pantheonplus, union3, dovekie")


# ============================================================================
# Theory
# ============================================================================

def get_theory():
    """Theory needed - wrapper gets cosmological parameters from CLASS"""
    
    return {
        "classy": {
            "path": None,
            "stop_at_error": True,
            "extra_args": {
                "N_ncdm": 1,
                "m_ncdm": 0.06,
                "deg_ncdm": 1,
                "T_ncdm": 0.71611,
                "N_ur": 2.0328,
                "output": "mPk",  # Minimal output for SN only
                "P_k_max_1/Mpc": 1.0,
                "Omega_Lambda": 0,  # Use w0wa instead
            },
        }
    }


# ============================================================================
# Sampler
# ============================================================================

def get_sampler():
    """MCMC sampler configuration"""
    
    return {
        "mcmc": {
            "Rminus1_stop": 0.01,
            "max_samples": 100000,
            "max_tries": 5000,
            "proposal_scale": 1.0,
            "learn_proposal": True,
            "burn_in": 0,
            "measure_speeds": False,
            "oversample_power": 0.2,
            "drag": False,  # No fast parameters in SN only
        }
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run single SN dataset analysis"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run single SN dataset MCMC (using desilike)"
    )
    parser.add_argument("--dataset", type=str, 
                       choices=["pantheonplus", "union3", "dovekie"],
                       help="SN dataset to use")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite")
    parser.add_argument("--resume", action="store_true",
                       help="Resume existing chains")
    
    args = parser.parse_args()
    
    # Determine dataset
    dataset = args.dataset if args.dataset else DATASET
    
    # Build configuration
    params = get_params()
    params.update(get_sn_params())
    
    likelihood = get_sn_likelihood(dataset)
    theory = get_theory()
    sampler = get_sampler()
    
    # Output directory
    output_dir = f"chains/sn_{dataset.lower()}_raw"
    
    # Build info
    info = {
        "params": params,
        "likelihood": likelihood,
        "theory": theory,
        "sampler": sampler,
        "output": output_dir,
        "force": args.force,
        "resume": args.resume,
        "debug": False,
    }
    
    # Dataset info
    dataset_names = {
        "pantheonplus": "PantheonPlus",
        "union3": "Union3",
        "dovekie": "DES-Dovekie"
    }
    
    dataset_nsn = {
        "pantheonplus": "~1701",
        "union3": "22 (UNITY1.5 format)",
        "dovekie": "1820"
    }
    
    dataset_likelihood = {
        "pantheonplus": "PantheonPlusSNLikelihood",
        "union3": "Union3SNLikelihood",
        "dovekie": "DESY5SNLikelihood"
    }
    
    # Print summary
    print("="*70)
    print(f"Single SN Dataset: {dataset_names[dataset.lower()]}")
    print("="*70)
    print(f"\nSetting:")
    print(f"  Dataset: {dataset_names[dataset.lower()]}")
    print(f"  SNe: {dataset_nsn[dataset.lower()]}")
    print(f"  Likelihood: {dataset_likelihood[dataset.lower()]} (desilike)")
    print(f"  Model: w0wa CDM")
    print(f"  Output: {output_dir}")
    
    # Count parameters
    n_cosmo = 5      # omega_b, omega_cdm, H0, w0_fld, wa_fld
    n_sn = 1         # Mb
    n_derived = 1    # Omega_m
    
    n_total = n_cosmo + n_sn
    
    print(f"\nParameters:")
    print(f"  Cosmology: {n_cosmo} (simplified w0wa)")
    print(f"  SN: {n_sn} (Mb)")
    print(f"  Derived: {n_derived} (Omega_m)")
    print(f"  Total sampling: {n_total}")
    
    print("\n✅ Fast SN only analysis - expect ~1-2 hours!")
    
    print("\n" + "="*70)
    print("Start MCMC...")
    print("="*70 + "\n")
    
    # Run
    updated_info, sampler_instance = run(info)
    
    print("\n" + "="*70)
    print("MCMC Complete!")
    print("="*70)
    print(f"\nResult: {output_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())