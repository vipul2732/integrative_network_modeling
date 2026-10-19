#!/usr/bin/env python3
"""
Minimal workflow runner for Bayesian network inference from APMS data.

Usage:
    python run.py --config config.yaml
    python run.py --model model23_n --data data.xlsx --output results/
"""

import os
import sys

# Must run before model_variations (and JAX) is imported below.
if '--cpu' in sys.argv:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

# Try to import yaml, fall back to json-only
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    logging.warning("PyYAML not available. Use JSON config files instead.")

# Add paths
script_dir = Path(__file__).parent
workflow_dir = script_dir.parent  # workflow/
project_root = workflow_dir.parent  # project root (cursor/)
sys.path.insert(0, str(project_root))  # Add project root for other modules
sys.path.append(str(project_root / "notebooks"))  # Add notebooks directory
import preprocess_data
# Import model_variations from local scripts folder (cleaned version)
import model_variations as mv

# Import figure generators (if available)
try:
    import generate_sampling_figures as gsf
    import generate_benchmark_figures as gbf
    import generate_analysis_figures as gaf
    HAS_FIGURES = True
except ImportError:
    HAS_FIGURES = False
    logging.warning("Figure generation modules not available")


def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    log_file = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML or JSON file, or return defaults."""
    if config_path and config_path.exists():
        with open(config_path) as f:
            if config_path.suffix in ['.yaml', '.yml']:
                if HAS_YAML:
                    config = yaml.safe_load(f)
                else:
                    raise ImportError("PyYAML required for YAML config files. Install with: pip install pyyaml")
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                # Try to auto-detect
                content = f.read()
                f.seek(0)
                try:
                    config = json.load(f)
                except json.JSONDecodeError:
                    if HAS_YAML:
                        config = yaml.safe_load(content)
                    else:
                        raise ValueError("Could not parse config file. Use JSON or install PyYAML for YAML support.")
        return config
    
    # Return sensible defaults
    return {
        'data': {
            'input_file': None,
            'prey_column': 'PreyGene',
            'filter': 'all',  # 'all', 'wt', 'vif', 'mock'
            'enforce_bait_remapping': True,
            'sheet_nums': 3,  # Number of sheets to process (None = all sheets)
        },
        'model': {
            'name': 'model23_n',
            'rseed': 13,
            'model_id': 0,
        },
        'sampling': {
            'num_warmup': 1000,
            'num_samples': 10000,
            'thinning': 1,
            'target_accept_prob': 0.8,
            'progress_bar': True,
        },
        'hyperparameters': {
            'alpha': 0.02,
            'beta': 0.1,
            'thresholds': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'n_null_bins': 80,
            'disconnectivity_distance': 10,
            'max_distance': 11,
            'use_indirect_saint': True,  # Default: True to enable indirect SAINT scores for prey-prey pairs
            'indirect_saint_threshold': 0.3,  # Minimum SAINT score for indirect prey-prey pairs (if enabled)
        },
        'output': {
            'output_dir': '../results/run',
            'save_warmup': True,
            'include_extra_fields': True,
        },
        'figures': {
            'generate': True,
        },
        'imp': {
            'generate': False,  # Set to True to generate IMP restraints
            'threshold': 0.7,    # Probability threshold for restraints
            'benchmark': False,  # Set to True to benchmark against PDB/CORUM
            'pdb_dir': None,     # Path to PDB structures directory
            'corum_file': None,  # Path to CORUM database file
        }
    }


def preprocess_data_step(config: dict, logger: logging.Logger) -> Path:
    """Step 1: Preprocess raw data."""
    data_cfg = config['data']
    output_cfg = config['output']
    
    # Resolve relative paths relative to script location
    input_file = Path(data_cfg['input_file'])
    if not input_file.is_absolute():
        # If relative, resolve from scripts/ directory
        script_dir = Path(__file__).parent
        input_file = (script_dir / input_file).resolve()
        # Also try resolving from workflow/input if not found
        if not input_file.exists():
            workflow_dir = script_dir.parent
            alt_path = workflow_dir / 'input' / input_file.name
            if alt_path.exists():
                input_file = alt_path
    
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create preprocessed data directory
    preprocessed_dir = Path(output_cfg['output_dir']) / 'preprocessed'
    if not preprocessed_dir.is_absolute():
        script_dir = Path(__file__).parent
        preprocessed_dir = (script_dir / preprocessed_dir).resolve()
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Preprocessing data from {input_file}")
    
    # Get sheet_nums from config, default to 3
    # If selected_sheet_indices is provided, use that (but note: preprocess_data
    # currently only supports consecutive sheets starting from 0)
    selected_indices = data_cfg.get('selected_sheet_indices', None)
    if selected_indices is not None:
        # Validate that selected sheets are consecutive starting from 0
        expected = list(range(len(selected_indices)))
        if selected_indices != expected:
            logger.warning(
                f"Selected sheet indices {selected_indices} are not consecutive starting from 0. "
                f"Preprocessing will use first {len(selected_indices)} sheets (0-{len(selected_indices)-1}). "
                f"To use non-consecutive sheets, manually edit the Excel file or modify preprocess_data.py"
            )
        sheet_nums = len(selected_indices)
    else:
        sheet_nums = data_cfg.get('sheet_nums', 3)
    
    logger.info(f"Processing {sheet_nums} sheet(s) from Excel file")
    if selected_indices:
        selected_names = data_cfg.get('selected_sheet_names', [])
        logger.info(f"Selected sheets: {', '.join(selected_names)}")
    
    preprocess_data.preprocess_spec_table(
        xlsx_path=str(input_file),
        output_dir=str(preprocessed_dir),
        sheet_nums=sheet_nums,
        prey_colname=data_cfg['prey_column'],
        enforce_bait_remapping=data_cfg['enforce_bait_remapping'],
        filter_kw=data_cfg['filter'],
        filter_zero_saint_scores=True,
    )
    
    logger.info(f"Preprocessed data saved to {preprocessed_dir}")
    return preprocessed_dir


def run_sampling_step(config: dict, preprocessed_dir: Path, logger: logging.Logger):
    """Step 2: Run MCMC sampling."""
    model_cfg = config['model']
    sampling_cfg = config['sampling']
    hyper_cfg = config['hyperparameters']
    output_cfg = config['output']
    
    output_dir = Path(output_cfg['output_dir'])
    if not output_dir.is_absolute():
        script_dir = Path(__file__).parent
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy preprocessed data to output directory
    shutil.copy(preprocessed_dir / 'spec_table.tsv', output_dir / 'spec_table.tsv')
    shutil.copy(preprocessed_dir / 'composite_table.tsv', output_dir / 'composite_table.tsv')
    
    # Save modeler variables
    modeler_vars = {
        'alpha': hyper_cfg['alpha'],
        'beta': hyper_cfg['beta'],
        'thresholds': hyper_cfg['thresholds'],
        'n_null_bins': hyper_cfg['n_null_bins'],
        'disconectivity_distance': hyper_cfg['disconnectivity_distance'],
        'max_distance': hyper_cfg['max_distance'],
        'use_indirect_saint': hyper_cfg.get('use_indirect_saint', True),  # Default: True (enabled since we filter to top 80 proteins)
        'indirect_saint_threshold': hyper_cfg.get('indirect_saint_threshold', 0.3),  # Default: 0.3 (lower threshold since dataset is smaller)
        # Reproducibility for data loader (correlation shuffle + random protein subset)
        'rseed': int(model_cfg['rseed']),
    }
    
    # Add hardcoded edge limits if provided (for quick test / performance)
    if 'hardcoded_expected_edges' in hyper_cfg:
        modeler_vars['hardcoded_expected_edges'] = hyper_cfg['hardcoded_expected_edges']
    if 'hardcoded_expected_edges_std' in hyper_cfg:
        modeler_vars['hardcoded_expected_edges_std'] = hyper_cfg['hardcoded_expected_edges_std']
    # Add max_edges for edge filtering (CRITICAL for performance)
    if 'max_edges' in hyper_cfg:
        modeler_vars['max_edges'] = hyper_cfg['max_edges']
    # Add protein filtering parameters (CRITICAL for performance - reduces N)
    # DEFAULT: Filter to top 80 proteins for all runs (can be overridden in config)
    # This dramatically speeds up JAX compilation: N² reduction from 177²=31,329 to 80²=6,400 (4.9x)
    if 'min_saint_score_for_protein' in hyper_cfg:
        modeler_vars['min_saint_score_for_protein'] = hyper_cfg['min_saint_score_for_protein']
    # Set max_proteins default to 80 if not specified (applies to ALL runs)
    max_proteins = hyper_cfg.get('max_proteins', 80)  # Default: 80 for all runs
    modeler_vars['max_proteins'] = max_proteins
    protein_selection = str(hyper_cfg.get('protein_selection', 'score')).strip().lower()
    modeler_vars['protein_selection'] = protein_selection
    if 'random_proteins_seed' in hyper_cfg and hyper_cfg['random_proteins_seed'] is not None:
        modeler_vars['random_proteins_seed'] = int(hyper_cfg['random_proteins_seed'])

    if protein_selection == 'random':
        eff = modeler_vars.get('random_proteins_seed', model_cfg['rseed'])
        if max_proteins is None or max_proteins <= 0:
            logger.info(
                "Protein selection: random — using **all** prey rows in spec_table (max_proteins null/0; "
                "no subsampling; no remove_prey / filter_minimal_ids). "
                "Expect very large N and long runs / high memory."
            )
        else:
            logger.info(
                f"Protein selection: random — uniformly sample up to {max_proteins} from full spec_table "
                f"(no remove_prey / filter_minimal_ids; seed={eff}; set random_proteins_seed to override)"
            )
    elif protein_selection == 'stratified_saint':
        eff = modeler_vars.get('random_proteins_seed', model_cfg['rseed'])
        logger.info(
            f"Protein selection: stratified_saint — same preprocess as base (remove_prey, filter_minimal_ids), "
            f"rank by combined SAINT score, then sample ~half of {max_proteins} from top-ranked half and ~half "
            f"from bottom-ranked half (seed={eff}; random_proteins_seed overrides subset RNG)"
        )
    elif 'max_proteins' not in hyper_cfg:
        logger.info(
            f"Using default protein filtering: top {max_proteins} proteins by combined score "
            f"(set 'max_proteins' / 'protein_selection' in config to override)"
        )
    else:
        logger.info(f"Protein filtering: top {max_proteins} proteins by combined score")
    
    with open(output_dir / 'modeler_vars.json', 'w') as f:
        json.dump(modeler_vars, f, indent=2)
    
    logger.info(f"Running model {model_cfg['name']} with {sampling_cfg['num_samples']} samples")
    
    # Run the model
    mv._main(
        model_id=model_cfg['model_id'],
        rseed=model_cfg['rseed'],
        model_name=model_cfg['name'],
        model_data=None,
        num_warmup=sampling_cfg['num_warmup'],
        num_samples=sampling_cfg['num_samples'],
        include_potential_energy=True,
        include_mean_accept_prob=False,
        include_extra_fields=output_cfg['include_extra_fields'],
        progress_bar=sampling_cfg['progress_bar'],
        save_dir=str(output_dir),
        initial_position=None,
        save_warmup=output_cfg['save_warmup'],
        load_warmup=False,
        jax_profile=False,
        init_strat='',
        thinning=sampling_cfg['thinning'],
        collect_warmup=False,
        synthetic_N=None,
        synthetic_Mtrue=None,
        synthetic_rseed=None,
    )
    
    logger.info(f"Sampling complete. Results saved to {output_dir}")


def generate_figures_step(config: dict, logger: logging.Logger):
    """Step 3: Generate figures."""
    if not config['figures']['generate'] or not HAS_FIGURES:
        logger.info("Skipping figure generation")
        return
    
    model_cfg = config['model']
    output_cfg = config['output']
    output_dir = Path(output_cfg['output_dir'])
    if not output_dir.is_absolute():
        script_dir = Path(__file__).parent
        output_dir = (script_dir / output_dir).resolve()
    
    fbasename = f"{model_cfg['model_id']}_{model_cfg['name']}_{model_cfg['rseed']}"
    input_file = output_dir / f"{fbasename}.pkl"
    
    if not input_file.exists():
        logger.warning(f"Results file not found: {input_file}. Skipping figures.")
        return
    
    logger.info("Generating figures...")
    
    try:
        gsf._main(
            o=str(output_dir),
            i=str(input_file),
            mode="cullin",
            merge=False
        )
        gbf.cullin_standard(output_dir, fbasename)
        gaf.cullin_standard()
        logger.info("Figures generated successfully")
    except Exception as e:
        logger.error(f"Error generating figures: {e}")


def generate_imp_restraints_step(config: dict, logger: logging.Logger):
    """Step 4: Generate IMP restraints from MCMC results (optional)."""
    if not config.get('imp', {}).get('generate', False):
        logger.info("Skipping IMP restraint generation")
        return
    
    try:
        import imp_integration
    except ImportError:
        logger.warning("imp_integration module not found. Skipping IMP restraints.")
        return
    
    model_cfg = config['model']
    output_cfg = config['output']
    imp_cfg = config.get('imp', {})
    
    output_dir = Path(output_cfg['output_dir'])
    if not output_dir.is_absolute():
        script_dir = Path(__file__).parent
        output_dir = (script_dir / output_dir).resolve()
    
    fbasename = f"{model_cfg['model_id']}_{model_cfg['name']}_{model_cfg['rseed']}"
    results_file = output_dir / f"{fbasename}.pkl"
    model_data_file = output_dir / f"{fbasename}_model_data.pkl"
    
    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}. Skipping IMP restraints.")
        return
    
    logger.info("Generating IMP restraints...")
    
    imp_output_dir = output_dir / 'imp_restraints'
    imp_output_dir.mkdir(exist_ok=True)
    
    # Use imp_integration module
    extractor = imp_integration.EdgeProbabilityExtractor(results_file, model_data_file)
    edge_probs, edge_dict = extractor.get_edge_probabilities()
    
    # Get node names
    node_names = list(extractor.node_idx2name.values()) if extractor.node_idx2name else \
                 [f"protein_{i}" for i in range(extractor.N)]
    
    # Save edge list
    threshold = imp_cfg.get('threshold', 0.7)
    extractor.save_edge_list(imp_output_dir / 'predicted_interactions.tsv', threshold=threshold)
    
    # Generate IMP restraints
    restraint_gen = imp_integration.IMPRestraintGenerator(edge_probs, node_names, edge_dict)
    restraint_gen.generate_restraints_script(
        imp_output_dir / 'imp_modeling_script.py',
        threshold=threshold
    )
    restraint_gen.save_restraints_xml(
        imp_output_dir / 'restraints.xml',
        threshold=threshold
    )
    
    # Benchmark if requested
    if imp_cfg.get('benchmark', False):
        logger.info("Running benchmarks...")
        benchmarker = imp_integration.Benchmarker(edge_dict)
        pdb_dir = imp_cfg.get('pdb_dir')
        corum_file = imp_cfg.get('corum_file')
        
        if pdb_dir:
            pdb_dir = Path(pdb_dir)
        if corum_file:
            corum_file = Path(corum_file)
        
        metrics_df = benchmarker.benchmark(pdb_dir=pdb_dir, corum_file=corum_file)
        
        if not metrics_df.empty:
            metrics_df.to_csv(imp_output_dir / 'benchmark_metrics.tsv', sep='\t', index=False)
            logger.info("Benchmark metrics saved")
    
    logger.info(f"IMP restraints saved to {imp_output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run Bayesian network inference workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file (single run)
  python run.py --config config.yaml
  
  # Override specific parameters
  python run.py --config config.yaml --model model23_n --samples 5000
  
  # Quick run with minimal options
  python run.py --data data.xlsx --model model23_n --output results/
  
  # Run with specific seed
  python run.py --config config.yaml --rseed 42
  
  # For multiple runs with different seeds, use run_multiple_seeds.py:
  python run_multiple_seeds.py --config config.yaml --n-runs 3
  python run_multiple_seeds.py --config config.yaml --seeds 13 42 99
        """
    )
    
    parser.add_argument('--config', type=Path, help='Path to YAML config file')
    parser.add_argument('--data', type=Path, help='Input Excel file path')
    parser.add_argument('--model', type=str, help='Model name (e.g., model23_n)')
    parser.add_argument('--output', type=Path, help='Output directory')
    parser.add_argument('--samples', type=int, help='Number of MCMC samples')
    parser.add_argument('--warmup', type=int, help='Number of warmup steps')
    parser.add_argument('--rseed', type=int, help='Random seed')
    parser.add_argument('--filter', type=str, choices=['all', 'wt', 'vif', 'mock'], help='Data filter')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force JAX on CPU (JAX_PLATFORMS=cpu). Slower but works without CUDA / busy GPUs.',
    )
    parser.add_argument('--multi-seed', type=int, nargs='+', metavar='SEED',
                       help='Run multiple times with different seeds (e.g., --multi-seed 13 42 99). '
                            'For more control, use run_multiple_seeds.py instead.')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.data:
        config['data']['input_file'] = str(args.data)
    if args.model:
        config['model']['name'] = args.model
    if args.output:
        config['output']['output_dir'] = str(args.output)
    if args.samples:
        config['sampling']['num_samples'] = args.samples
    if args.warmup:
        config['sampling']['num_warmup'] = args.warmup
    if args.rseed:
        config['model']['rseed'] = args.rseed
    if args.filter:
        config['data']['filter'] = args.filter
    if args.no_figures:
        config['figures']['generate'] = False
    
    # Validate required parameters
    if not config['data']['input_file']:
        parser.error("--data is required if not specified in config file")
    
    # Setup
    output_dir = Path(config['output']['output_dir'])
    if not output_dir.is_absolute():
        # If relative, resolve from scripts/ directory
        script_dir = Path(__file__).parent
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir)
    
    # Save final config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Starting workflow")
    logger.info("=" * 60)
    
    try:
        # Step 1: Preprocess
        preprocessed_dir = preprocess_data_step(config, logger)
        
        # Step 2: Run sampling
        run_sampling_step(config, preprocessed_dir, logger)
        
        # Step 3: Generate figures
        generate_figures_step(config, logger)
        
        # Step 4: Generate IMP restraints (optional)
        generate_imp_restraints_step(config, logger)
        
        logger.info("=" * 60)
        logger.info("Workflow complete!")
        logger.info(f"Results: {output_dir}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
