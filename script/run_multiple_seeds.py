#!/usr/bin/env python3
"""
Run the workflow multiple times with different random seeds.

This script:
1. Runs the workflow N times with different seeds
2. Saves results in separate directories
3. Prepares data for merging and convergence analysis

Usage:
    python run_multiple_seeds.py --config config_my_dataset.yaml --n-runs 3 --seeds 13 42 99
    python run_multiple_seeds.py --config config_my_dataset.yaml --n-runs 3  # Auto-generates seeds
"""

import argparse
import json
import logging
import os
import resource
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

# Import the main run script
import run

def setup_logging(output_dir: Path):
    """Setup logging."""
    log_file = output_dir / "multi_seed_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_single_seed(
    config_path: Path,
    seed: int,
    output_base: Path,
    logger: logging.Logger,
    test_mode: bool = False,
    use_cpu: bool = False,
):
    """Run workflow for a single seed."""
    seed_output_dir = output_base / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info(f"Running with seed: {seed}")
    logger.info(f"Output directory: {seed_output_dir}")
    if test_mode:
        logger.info("TEST MODE: Using minimal samples (100 warmup, 500 samples)")
    logger.info(f"=" * 60)
    
    # Build command with Python faulthandler for better crash diagnostics
    cmd = [
        sys.executable,
        "-X", "faulthandler",  # Enable faulthandler for segfault diagnostics
        "run.py",
        "--config", str(config_path),
        "--rseed", str(seed),
        "--output", str(seed_output_dir),
        "--no-figures"  # Skip figures for faster runs
    ]
    
    if test_mode:
        cmd.extend(["--warmup", "100", "--samples", "500"])
    if use_cpu:
        cmd.append("--cpu")
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    if use_cpu:
        env["JAX_PLATFORMS"] = "cpu"
    # Increase stack size to prevent segfaults during JAX compilation
    env['JAX_STACK_SIZE'] = '8388608'  # 8MB stack
    # Limit JAX memory preallocation to prevent OOM
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
    # Enable JAX debugging (slower but more informative)
    # env['JAX_ENABLE_X64'] = 'true'  # Use float64 (more stable but slower)
    
    # Try to increase stack size limit (if possible)
    try:
        # Get current stack limit
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        # Try to set to 64MB (minimum recommended for JAX)
        new_soft = min(64 * 1024 * 1024, hard)  # 64MB or hard limit, whichever is smaller
        resource.setrlimit(resource.RLIMIT_STACK, (new_soft, hard))
        logger.info(f"Stack size limit: {soft / 1024 / 1024:.1f}MB → {new_soft / 1024 / 1024:.1f}MB")
    except (ValueError, OSError) as e:
        logger.warning(f"Could not increase stack size limit: {e}")
    
    try:
        # Use Popen to stream output in real-time
        # Capture stderr separately to see segfault details
        process = subprocess.Popen(
            cmd,
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture stderr separately
            text=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        # Stream output in real-time
        logger.info(f"Starting seed {seed} (PID: {process.pid})...")
        logger.info("=" * 60)
        
        # Use threading to capture both stdout and stderr
        stderr_lines = []
        
        def read_stderr():
            for line in process.stderr:
                stderr_lines.append(line)
                # Also print stderr for visibility
                print(f"[Seed {seed} STDERR] {line.rstrip()}", flush=True)
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        for line in process.stdout:
            line = line.rstrip()
            if line:
                # Print to both logger and stdout for visibility
                print(f"[Seed {seed}] {line}", flush=True)
                logger.info(f"[Seed {seed}] {line}")
        
        process.wait()
        stderr_thread.join(timeout=1.0)  # Wait for stderr to finish
        
        logger.info("=" * 60)
        
        if process.returncode == 0:
            logger.info(f"✓ Seed {seed} completed successfully")
            return True, seed_output_dir
        else:
            logger.error(f"✗ Seed {seed} failed with exit code {process.returncode}")
            if stderr_lines:
                logger.error(f"STDERR output from seed {seed}:")
                for line in stderr_lines[-20:]:  # Last 20 lines
                    logger.error(f"  {line.rstrip()}")
            # Save stderr to file for debugging
            stderr_file = seed_output_dir / "stderr.log"
            with open(stderr_file, 'w') as f:
                f.writelines(stderr_lines)
            logger.error(f"Full stderr saved to: {stderr_file}")
            return False, seed_output_dir
            
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Seed {seed} failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False, seed_output_dir
    except Exception as e:
        logger.error(f"✗ Seed {seed} failed with error: {e}")
        return False, seed_output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Run workflow multiple times with different seeds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 3 times with auto-generated seeds
  python run_multiple_seeds.py --config config_my_dataset.yaml --n-runs 3
  
  # Run with specific seeds
  python run_multiple_seeds.py --config config_my_dataset.yaml --seeds 13 42 99
  
  # Run with custom output directory
  python run_multiple_seeds.py --config config_my_dataset.yaml --n-runs 3 --output-dir results/multi_seed_runs
        """
    )
    
    parser.add_argument('--config', type=Path, required=True, help='Path to config file')
    parser.add_argument('--n-runs', type=int, help='Number of runs (if not specifying seeds)')
    parser.add_argument('--seeds', type=int, nargs='+', help='Specific seed values to use')
    parser.add_argument('--output-dir', type=Path, help='Base output directory for all runs')
    parser.add_argument('--parallel', action='store_true', help='Run seeds in parallel (not implemented yet)')
    parser.add_argument('--test', action='store_true', help='Run quick test with minimal samples (100 warmup, 500 samples)')
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force JAX on CPU for every seed (JAX_PLATFORMS=cpu; passes --cpu to run.py).',
    )
    
    args = parser.parse_args()
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.n_runs:
        # Auto-generate seeds starting from 13
        seeds = [13 + i * 29 for i in range(args.n_runs)]  # 13, 42, 71, ...
    else:
        parser.error("Must specify either --n-runs or --seeds")
    
    # Resolve config path to absolute (important for run.py which runs from script/ directory)
    script_dir = Path(__file__).parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to script directory first
        if (script_dir / config_path).exists():
            config_path = (script_dir / config_path).resolve()
        # Try relative to project root
        elif (script_dir.parent / config_path).exists():
            config_path = (script_dir.parent / config_path).resolve()
        else:
            # Try as-is (might be relative to current working directory)
            config_path = config_path.resolve()
    
    if not config_path.exists():
        parser.error(f"Config file not found: {config_path}")
    
    # Determine output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
        if not output_base.is_absolute():
            output_base = (script_dir / output_base).resolve()
    else:
        # Default: use config name + _multi_seeds
        config_name = config_path.stem
        output_base = script_dir.parent / "results" / f"{config_name}_multi_seeds"
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_base)
    
    logger.info("=" * 60)
    logger.info("MULTI-SEED WORKFLOW RUNNER")
    logger.info("=" * 60)
    logger.info(f"Config: {config_path}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Number of runs: {len(seeds)}")
    if args.cpu:
        logger.info("JAX backend: CPU (--cpu)")
    logger.info("=" * 60)
    
    # Save run configuration
    run_config = {
        'config_file': str(config_path),
        'seeds': seeds,
        'output_base': str(output_base),
        'n_runs': len(seeds)
    }
    with open(output_base / 'multi_seed_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
    
    # Run each seed
    results = []
    for i, seed in enumerate(seeds, 1):
        logger.info(f"\n[{i}/{len(seeds)}] Starting run with seed {seed}")
        success, output_dir = run_single_seed(
            config_path,
            seed,
            output_base,
            logger,
            test_mode=args.test,
            use_cpu=args.cpu,
        )
        results.append({
            'seed': seed,
            'success': success,
            'output_dir': str(output_dir)
        })
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"Successful runs: {len(successful)}/{len(results)}")
    logger.info(f"Failed runs: {len(failed)}/{len(results)}")
    
    if successful:
        logger.info(f"\nSuccessful seeds: {[r['seed'] for r in successful]}")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Merge results: python merge_mcmc_results.py {output_base}")
        logger.info(f"  2. Analyze convergence: python analyze_convergence.py {output_base}")
    
    if failed:
        logger.warning(f"\nFailed seeds: {[r['seed'] for r in failed]}")
    
    # Save results summary
    with open(output_base / 'run_summary.json', 'w') as f:
        json.dump({
            'results': results,
            'successful_count': len(successful),
            'failed_count': len(failed)
        }, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_base}")
    logger.info("=" * 60)
    
    return 0 if len(failed) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())

