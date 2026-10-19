#!/usr/bin/env python3
"""
Run ablated models for robustness testing.

This script runs three ablated versions of model23_n:
1. model23_n_no_corr: Without correlation constraints
2. model23_n_no_saint: Without SAINT score constraint
3. model23_n_no_priors: Without degree and edge count priors

Usage:
    python run_ablated_models.py \
        --config script/configs/cullin_mock_config.json \
        --base-dataset cullin_mock_config_multi_seeds \
        --seeds 13 42 100 200 300
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration file."""
    # Resolve path relative to script directory if needed
    if not config_path.is_absolute():
        # Try relative to current working directory first
        if not config_path.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent
            alt_path = script_dir / config_path
            if alt_path.exists():
                config_path = alt_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")


def create_ablated_config(base_config: dict, model_name: str, output_dir: str) -> dict:
    """Create config for ablated model."""
    ablated_config = base_config.copy()
    ablated_config['model']['name'] = model_name
    ablated_config['output']['output_dir'] = output_dir
    return ablated_config


def run_single_ablated_model(
    base_config_path: Path,
    model_name: str,
    dataset_suffix: str,
    seeds: list,
    base_results_dir: Path
):
    """Run a single ablated model with multiple seeds."""
    logger.info("=" * 60)
    logger.info(f"Running ablated model: {model_name}")
    logger.info("=" * 60)
    
    # Get script directory (project root)
    script_dir = Path(__file__).parent.parent
    
    # Load base config
    base_config = load_config(base_config_path)
    
    # Create output directory name - infer from config file path
    base_config_path_resolved = Path(base_config_path)
    if not base_config_path_resolved.is_absolute():
        base_config_path_resolved = script_dir / base_config_path_resolved
    
    # Infer dataset name from config file name
    config_stem = base_config_path_resolved.stem
    if 'cullin_mock' in config_stem:
        base_dataset = 'cullin_mock'
    elif 'cullin_wt' in config_stem:
        base_dataset = 'cullin_wt'
    elif 'cullin_vif' in config_stem:
        base_dataset = 'cullin_vif'
    else:
        # Try metadata or use stem
        base_dataset = base_config.get('metadata', {}).get('dataset_name', config_stem.replace('_config', ''))
    
    # Remove _multi_seeds suffix if present
    base_dataset = base_dataset.replace('_multi_seeds', '').replace('_config', '')
    output_dir = f"{base_dataset}_{dataset_suffix}"
    # Resolve to absolute path since run_multiple_seeds.py runs from script/ directory
    if not base_results_dir.is_absolute():
        base_results_dir = (script_dir / base_results_dir).resolve()
    output_path = base_results_dir / output_dir
    
    # Create config for ablated model
    ablated_config = create_ablated_config(base_config, model_name, f"../results/{output_dir}")
    
    # Save ablated config (in same directory as base config)
    config_dir = base_config_path_resolved.parent
    ablated_config_path = config_dir / f"config_{dataset_suffix}.json"
    with open(ablated_config_path, 'w') as f:
        json.dump(ablated_config, f, indent=2)
    
    logger.info(f"Created ablated config: {ablated_config_path}")
    logger.info(f"Output directory: {output_path}")
    
    # Run with multiple seeds using run_multiple_seeds.py
    script_dir = Path(__file__).parent.parent
    cmd = [
        'python', str(script_dir / 'script' / 'run_multiple_seeds.py'),
        '--config', str(ablated_config_path),
        '--seeds'
    ]
    # Add seeds as separate arguments (not as a single string)
    cmd.extend([str(s) for s in seeds])
    # Note: run_multiple_seeds.py uses --output-dir, not --results-dir
    # output_path is already absolute, so pass it directly
    cmd.extend(['--output-dir', str(output_path)])
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Set environment variables for JAX (use CPU if CUDA not available)
    env = os.environ.copy()
    env['JAX_PLATFORMS'] = 'cpu'  # Force CPU to avoid CUDA initialization errors
    
    result = subprocess.run(cmd, cwd=script_dir, env=env)
    
    if result.returncode != 0:
        logger.error(f"Failed to run {model_name}")
        return False
    
    logger.info(f"Successfully completed {model_name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run ablated models for robustness testing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all ablated models with default seeds
  python run_ablated_models.py \\
    --config script/configs/cullin_mock_config.json \\
    --base-dataset cullin_mock_config_multi_seeds

  # Run with custom seeds
  python run_ablated_models.py \\
    --config script/configs/cullin_mock_config.json \\
    --base-dataset cullin_mock_config_multi_seeds \\
    --seeds 13 42 100 200 300
        """
    )
    
    parser.add_argument('--config', type=Path, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--base-dataset', type=str, required=True,
                       help='Base dataset name (e.g., cullin_mock_config_multi_seeds)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[13, 42, 100, 200, 300],
                       help='Random seeds to use (default: 13 42 100 200 300)')
    parser.add_argument('--results-dir', type=Path, default=Path('../results'),
                       help='Base results directory (default: ../results)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip models that already have results')
    
    args = parser.parse_args()
    
    # Define ablated models
    ablated_models = [
        ('model23_n_no_corr', 'no_corr', 'Without correlation constraints'),
        ('model23_n_no_saint', 'no_saint', 'Without SAINT score constraint'),
        ('model23_n_no_priors', 'no_priors', 'Without degree and edge count priors'),
    ]
    
    logger.info("=" * 60)
    logger.info("Running Ablated Models for Robustness Testing")
    logger.info("=" * 60)
    logger.info(f"Base config: {args.config}")
    logger.info(f"Base dataset: {args.base_dataset}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Results dir: {args.results_dir}")
    logger.info("")
    
    # Run each ablated model
    results = {}
    for model_name, suffix, description in ablated_models:
        logger.info(f"\n{description}")
        logger.info(f"Model: {model_name}")
        
        # Check if results already exist
        # Extract base dataset name (remove _multi_seeds and _config if present)
        base_dataset_clean = args.base_dataset.replace('_multi_seeds', '').replace('_config', '')
        output_dir = f"{base_dataset_clean}_{suffix}"
        output_path = args.results_dir / output_dir
        
        if args.skip_existing and output_path.exists():
            logger.info(f"Skipping {model_name} - results already exist at {output_path}")
            results[model_name] = True
            continue
        
        success = run_single_ablated_model(
            args.config,
            model_name,
            suffix,
            args.seeds,
            args.results_dir
        )
        results[model_name] = success
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for model_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{model_name:25s}: {status}")
    
    # Check if all succeeded
    all_success = all(results.values())
    if not all_success:
        logger.error("\nSome models failed. Check logs above.")
        sys.exit(1)
    
    logger.info("\nAll ablated models completed successfully!")
    logger.info("\nNext steps:")
    logger.info("  1. Merge chains: python script/merge_and_analyze_convergence.py ...")
    logger.info("  2. Compare results: python script/compare_ablated_models.py ...")


if __name__ == '__main__':
    main()
