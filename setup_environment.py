#!/usr/bin/env python3
"""
Python-based setup script for Integrative Network Modeling (INM) environment.
Alternative to the bash script for cross-platform compatibility.

Usage:
    python setup_environment.py
    python setup_environment.py --env-name my_env --python-version 3.10
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, check=True, shell=False):
    """Run a shell command and return success status."""
    try:
        if isinstance(cmd, str) and shell:
            result = subprocess.run(cmd, shell=True, check=check, 
                                  capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, 
                                  capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}", file=sys.stderr)
        return False


def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run(['conda', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Conda found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("ERROR: conda is not installed or not in PATH")
    print("Please install Miniconda or Anaconda first:")
    print("  https://docs.conda.io/en/latest/miniconda.html")
    return False


def check_env_exists(env_name):
    """Check if conda environment exists."""
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        return env_name in result.stdout
    except:
        return False


def create_environment(env_name, python_version, force=False):
    """Create conda environment."""
    if check_env_exists(env_name):
        if not force:
            print(f"\nWARNING: Environment '{env_name}' already exists!")
            response = input("Do you want to remove it and create a fresh one? (y/N): ")
            if response.lower() != 'y':
                print(f"Keeping existing environment. Activate with: conda activate {env_name}")
                return False
            print("Removing existing environment...")
            run_command(['conda', 'env', 'remove', '-n', env_name, '-y'])
        else:
            run_command(['conda', 'env', 'remove', '-n', env_name, '-y'])
    
    print(f"\nCreating conda environment '{env_name}' with Python {python_version}...")
    return run_command(['conda', 'create', '-n', env_name, f'python={python_version}', '-y'])


def install_conda_packages(env_name):
    """Install packages via conda."""
    print("\nInstalling conda packages...")
    return run_command(['conda', 'install', '-n', env_name, 
                       '-c', 'conda-forge', 'numpy', 'scipy', '-y'])


def install_pip_packages(env_name, use_gpu=False, cuda_version='12'):
    """Install packages via pip."""
    print("\nInstalling pip packages...")
    
    # Get pip path for the environment
    conda_base = subprocess.run(['conda', 'info', '--base'], 
                               capture_output=True, text=True).stdout.strip()
    pip_path = Path(conda_base) / 'envs' / env_name / 'bin' / 'pip'
    
    if not pip_path.exists():
        print(f"ERROR: pip not found at {pip_path}")
        return False
    
    # Upgrade pip
    run_command([str(pip_path), 'install', '--upgrade', 'pip'])
    
    # Core packages
    packages = [
        'numpyro>=0.18.0',
        'arviz>=0.23.0',
        'pandas>=2.0.0',
        'matplotlib>=3.8.0',
        'seaborn>=0.13.0',
        'networkx>=3.0',
        'openpyxl>=3.1.0',
        'pyyaml>=6.0',
        'click>=8.0',
        'python-louvain>=0.16',
        'scikit-learn>=1.3.0',
    ]
    
    # JAX installation
    if use_gpu:
        if cuda_version == '12':
            packages.insert(2, 'jax[cuda12]')
            packages.insert(3, 'jax-cuda12-plugin')
        elif cuda_version == '11':
            packages.insert(2, 'jax[cuda11]')
            packages.insert(3, 'jax-cuda11-plugin')
        else:
            packages.insert(2, 'jax>=0.6.0')
            packages.insert(3, 'jaxlib>=0.6.0')
    else:
        packages.insert(2, 'jax>=0.6.0')
        packages.insert(3, 'jaxlib>=0.6.0')
    
    # Install all packages
    cmd = [str(pip_path), 'install'] + packages
    return run_command(cmd)


def verify_installation(env_name):
    """Verify that key packages are installed."""
    print("\n" + "="*50)
    print("Verifying installation...")
    print("="*50)
    
    conda_base = subprocess.run(['conda', 'info', '--base'], 
                               capture_output=True, text=True).stdout.strip()
    python_path = Path(conda_base) / 'envs' / env_name / 'bin' / 'python'
    
    if not python_path.exists():
        print(f"ERROR: Python not found at {python_path}")
        return False
    
    checks = [
        ('numpyro', 'numpyro.__version__'),
        ('jax', 'jax.__version__'),
        ('numpy', 'numpy.__version__'),
        ('pandas', 'pandas.__version__'),
        ('matplotlib', 'matplotlib.__version__'),
        ('networkx', 'networkx.__version__'),
    ]
    
    all_ok = True
    for package, version_expr in checks:
        try:
            cmd = [str(python_path), '-c', f'import {package}; print("{package}:", {version_expr})']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ {result.stdout.strip()}")
            else:
                print(f"✗ {package}: Import failed")
                all_ok = False
        except Exception as e:
            print(f"✗ {package}: {e}")
            all_ok = False
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description='Setup INM conda environment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--env-name', default='inm-py310',
                       help='Conda environment name (default: inm-py310)')
    parser.add_argument('--python-version', default='3.10',
                       help='Python version (default: 3.10)')
    parser.add_argument('--force', action='store_true',
                       help='Force recreate environment if it exists')
    parser.add_argument('--gpu', action='store_true',
                       help='Install JAX with GPU support')
    parser.add_argument('--cuda-version', choices=['11', '12'], default='12',
                       help='CUDA version for GPU support (default: 12)')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip installation verification')
    
    args = parser.parse_args()
    
    print("="*50)
    print("INM Environment Setup (Python)")
    print("="*50)
    print()
    
    # Check conda
    if not check_conda():
        sys.exit(1)
    
    # Create environment
    if not create_environment(args.env_name, args.python_version, args.force):
        sys.exit(1)
    
    # Install conda packages
    if not install_conda_packages(args.env_name):
        print("WARNING: Some conda packages may have failed to install")
    
    # Ask about GPU if not specified
    use_gpu = args.gpu
    if not use_gpu and not args.gpu:
        response = input("\nDo you have a CUDA-capable GPU? (y/N): ")
        use_gpu = response.lower() == 'y'
        if use_gpu:
            cuda_response = input("CUDA version? (11 or 12, default: 12): ")
            if cuda_response in ['11', '12']:
                args.cuda_version = cuda_response
    
    # Install pip packages
    if not install_pip_packages(args.env_name, use_gpu, args.cuda_version):
        print("ERROR: Failed to install pip packages")
        sys.exit(1)
    
    # Verify installation
    if not args.skip_verify:
        verify_installation(args.env_name)
    
    print("\n" + "="*50)
    print("Setup Complete!")
    print("="*50)
    print(f"\nTo activate the environment, run:")
    print(f"  conda activate {args.env_name}")
    print()
    if use_gpu:
        print("To verify JAX can see your GPU, run:")
        print(f"  conda activate {args.env_name}")
        print("  python -c 'import jax; print(jax.devices())'")
        print()


if __name__ == '__main__':
    main()
