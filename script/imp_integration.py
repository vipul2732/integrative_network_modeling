#!/usr/bin/env python3
"""
IMP Integration Module for Bayesian Network Inference

This module:
1. Extracts edge probabilities from MCMC samples
2. Converts them to IMP restraints (distance, connectivity, excluded volume)
3. Provides benchmarking against PDB and CORUM databases
4. Generates IMP scripts for complex modeling

Usage:
    python imp_integration.py --results results/0_model23_n_13.pkl --output imp_restraints/
    python imp_integration.py --results results/0_model23_n_13.pkl --benchmark --pdb-dir data/pdb/
"""

import argparse
import json
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

try:
    import IMP
    import IMP.atom
    import IMP.core
    import IMP.algebra
    import IMP.container
    HAS_IMP = True
except ImportError:
    HAS_IMP = False
    logging.warning("IMP not available. Install with: conda install -c salilab imp")

import jax.numpy as jnp
from model_variations import Z2A, ij_from, node_list2edge_list

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class EdgeProbabilityExtractor:
    """Extract edge probabilities from MCMC samples."""
    
    def __init__(self, results_file: Path, model_data_file: Optional[Path] = None):
        """
        Initialize extractor with MCMC results.
        
        Args:
            results_file: Path to MCMC results pickle file
            model_data_file: Optional path to model_data pickle file (for name mappings)
        """
        self.results_file = Path(results_file)
        self.model_data_file = model_data_file
        
        # Load results
        with open(self.results_file, 'rb') as f:
            self.results = pkl.load(f)
        
        self.samples = self.results.get('samples', {})
        self.extra_fields = self.results.get('extra_fields', {})
        
        # Load model data if available
        if model_data_file and Path(model_data_file).exists():
            with open(model_data_file, 'rb') as f:
                self.model_data = pkl.load(f)
            self.name2node_idx = self.model_data.get('name2node_idx', {})
            self.node_idx2name = self.model_data.get('node_idx2name', {})
        else:
            # Try to find model_data file in same directory
            model_data_pattern = str(self.results_file).replace('.pkl', '_model_data.pkl')
            if Path(model_data_pattern).exists():
                with open(model_data_pattern, 'rb') as f:
                    self.model_data = pkl.load(f)
                self.name2node_idx = self.model_data.get('name2node_idx', {})
                self.node_idx2name = self.model_data.get('node_idx2name', {})
            else:
                logger.warning("Model data file not found. Will use indices instead of names.")
                self.model_data = {}
                self.name2node_idx = {}
                self.node_idx2name = {}
        
        # Get N from samples
        if 'z' in self.samples:
            z_shape = self.samples['z'].shape
            # z is flattened edge list: M = N*(N-1)//2
            # Solve: N*(N-1)/2 = M
            M = z_shape[-1] if len(z_shape) > 1 else len(self.samples['z'])
            # N = (1 + sqrt(1 + 8*M)) / 2
            self.N = int((1 + np.sqrt(1 + 8 * M)) / 2)
            self.M = M
        else:
            raise ValueError("No 'z' samples found in results")
    
    def get_edge_probabilities(self, threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """
        Extract edge probabilities from MCMC samples.
        
        Args:
            threshold: Probability threshold for binary edges
            
        Returns:
            edge_probs: Matrix of edge probabilities (N x N)
            edge_dict: Dictionary mapping (i, j) -> probability
        """
        if 'z' not in self.samples:
            raise ValueError("No 'z' samples found")
        
        z_samples = self.samples['z']
        
        # Convert z to numpy if JAX array
        if hasattr(z_samples, 'block_until_ready'):
            z_samples = np.array(z_samples)
        
        # Handle multiple chains: shape might be (n_chains, n_samples, M) or (n_samples, M)
        if len(z_samples.shape) == 3:
            # Multiple chains: flatten to (n_chains * n_samples, M)
            z_samples = z_samples.reshape(-1, z_samples.shape[-1])
        elif len(z_samples.shape) == 2:
            # Single chain: (n_samples, M)
            pass
        else:
            # Single sample: (M,)
            z_samples = z_samples.reshape(1, -1)
        
        n_samples = z_samples.shape[0]
        logger.info(f"Processing {n_samples} MCMC samples")
        
        # Convert z to binary adjacency matrices and compute probabilities
        adjacency_sum = np.zeros((self.N, self.N))
        
        for i in range(n_samples):
            z = z_samples[i]
            # Convert z to adjacency matrix
            aij = Z2A(jnp.array(z), self.N)
            aij = np.array(aij)
            adjacency_sum += aij
        
        # Compute edge probabilities
        edge_probs = adjacency_sum / n_samples
        
        # Create edge dictionary with names if available
        edge_dict = {}
        for i in range(self.N):
            for j in range(i + 1, self.N):
                prob = edge_probs[i, j]
                if prob > 0:  # Only store non-zero probabilities
                    name_i = self.node_idx2name.get(i, f"protein_{i}")
                    name_j = self.node_idx2name.get(j, f"protein_{j}")
                    edge_dict[(name_i, name_j)] = float(prob)
        
        return edge_probs, edge_dict
    
    def get_high_confidence_edges(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        Get high-confidence edges above threshold.
        
        Args:
            threshold: Minimum probability threshold
            
        Returns:
            List of (protein1, protein2, probability) tuples
        """
        _, edge_dict = self.get_edge_probabilities()
        
        high_conf_edges = [
            (p1, p2, prob) for (p1, p2), prob in edge_dict.items()
            if prob >= threshold
        ]
        
        # Sort by probability (descending)
        high_conf_edges.sort(key=lambda x: x[2], reverse=True)
        
        return high_conf_edges
    
    def save_edge_list(self, output_file: Path, threshold: float = 0.5):
        """Save edge list to TSV file."""
        _, edge_dict = self.get_edge_probabilities()
        
        edges = []
        for (p1, p2), prob in edge_dict.items():
            if prob >= threshold:
                edges.append({
                    'Protein1': p1,
                    'Protein2': p2,
                    'Probability': prob,
                    'Confidence': 'High' if prob >= 0.7 else 'Medium' if prob >= 0.5 else 'Low'
                })
        
        df = pd.DataFrame(edges)
        df = df.sort_values('Probability', ascending=False)
        df.to_csv(output_file, sep='\t', index=False)
        logger.info(f"Saved {len(df)} edges to {output_file}")


class IMPRestraintGenerator:
    """Generate IMP restraints from edge probabilities."""
    
    def __init__(self, edge_probs: np.ndarray, node_names: List[str], 
                 edge_dict: Optional[Dict] = None):
        """
        Initialize restraint generator.
        
        Args:
            edge_probs: N x N matrix of edge probabilities
            node_names: List of protein names (length N)
            edge_dict: Optional dictionary mapping (name1, name2) -> probability
        """
        self.edge_probs = edge_probs
        self.node_names = node_names
        self.edge_dict = edge_dict or {}
        self.N = len(node_names)
    
    def create_distance_restraint(self, model: 'IMP.Model', p1: 'IMP.Particle', 
                                  p2: 'IMP.Particle', distance: float, 
                                  weight: float = 1.0, sigma: float = 5.0):
        """
        Create a distance restraint between two particles.
        
        Args:
            model: IMP model
            p1, p2: IMP particles
            distance: Target distance (Angstroms)
            weight: Restraint weight
            sigma: Uncertainty (Angstroms)
        """
        d = IMP.core.DistanceRestraint(
            model,
            IMP.core.Harmonic(distance, sigma),
            p1, p2
        )
        d.set_weight(weight)
        return d
    
    def create_connectivity_restraint(self, model: 'IMP.Model', 
                                     connected_pairs: List[Tuple[str, str]],
                                     weight: float = 1.0):
        """
        Create connectivity restraint ensuring connected pairs are close.
        
        Args:
            model: IMP model
            connected_pairs: List of (protein1, protein2) pairs that should be connected
            weight: Restraint weight
        """
        # This would use IMP's connectivity scoring
        # For now, return a placeholder
        logger.info(f"Creating connectivity restraint for {len(connected_pairs)} pairs")
        return None
    
    def generate_restraints_script(self, output_file: Path, 
                                   threshold: float = 0.7,
                                   max_distance: float = 50.0,
                                   min_distance: float = 10.0):
        """
        Generate Python script for IMP modeling with restraints.
        
        Args:
            output_file: Path to output Python script
            threshold: Minimum edge probability for restraint
            max_distance: Maximum distance for connected pairs (Angstroms)
            min_distance: Minimum distance (excluded volume, Angstroms)
        """
        if not HAS_IMP:
            logger.error("IMP not available. Cannot generate restraints script.")
            return
        
        script_content = f'''#!/usr/bin/env python3
"""
IMP modeling script with Bayesian network restraints.
Generated from MCMC edge probabilities.

Usage:
    python {output_file.name} --pdb-dir data/pdb/ --output models/
"""

import IMP
import IMP.atom
import IMP.core
import IMP.algebra
import IMP.container
import sys
from pathlib import Path

# Initialize IMP
IMP.set_log_level(IMP.SILENT)
model = IMP.Model()

# Create particles for each protein
particles = {{}}
protein_names = {self.node_names}

for name in protein_names:
    p = IMP.Particle(model, name)
    # Add coordinates (will be optimized)
    IMP.core.XYZ.setup_particle(p, IMP.algebra.Vector3D(0, 0, 0))
    particles[name] = p

# Distance restraints from high-confidence edges
restraints = []
high_confidence_edges = [
'''
        
        # Add high-confidence edges
        high_conf = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                prob = self.edge_probs[i, j]
                if prob >= threshold:
                    name_i = self.node_names[i]
                    name_j = self.node_names[j]
                    # Convert probability to distance (higher prob = closer)
                    # Use inverse relationship: distance = max_dist * (1 - prob)
                    distance = max_distance * (1 - prob) + min_distance
                    weight = prob  # Higher probability = stronger restraint
                    high_conf.append((name_i, name_j, distance, weight))
        
        for name_i, name_j, distance, weight in high_conf:
            script_content += f"    ('{name_i}', '{name_j}', {distance:.2f}, {weight:.3f}),\n"
        
        script_content += f''']
        
# Create distance restraints
for name1, name2, distance, weight in high_confidence_edges:
    p1 = particles[name1]
    p2 = particles[name2]
    d = IMP.core.DistanceRestraint(
        model,
        IMP.core.Harmonic(distance, 5.0),  # 5 Angstrom uncertainty
        p1, p2
    )
    d.set_weight(weight)
    restraints.append(d)
    model.add_restraint(d)

# Excluded volume restraints (all pairs should not be too close)
excluded_volume = IMP.core.ExcludedVolumeRestraint(
    IMP.container.ListPairContainer(
        model, [particles[name] for name in protein_names]
    ),
    1.0  # Minimum distance
)
model.add_restraint(excluded_volume)

# Create scoring function
sf = IMP.core.RestraintsScoringFunction(restraints + [excluded_volume])

print(f"Created {{len(restraints)}} distance restraints")
print(f"Created 1 excluded volume restraint")
print(f"Total restraints: {{len(restraints) + 1}}")

# Save model state
model_state = {{
    'particles': particles,
    'restraints': restraints,
    'scoring_function': sf,
    'model': model
}}

# Example: Optimize structure
# optimizer = IMP.core.ConjugateGradients(model)
# optimizer.set_scoring_function(sf)
# optimizer.optimize(1000)

print("Model setup complete. Use IMP optimization to refine structure.")
'''
        
        with open(output_file, 'w') as f:
            f.write(script_content)
        
        logger.info(f"Generated IMP restraints script: {output_file}")
        logger.info(f"  High-confidence edges: {len(high_conf)}")
    
    def save_restraints_xml(self, output_file: Path, threshold: float = 0.7):
        """Save restraints in XML format for IMP."""
        high_conf_edges = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                prob = self.edge_probs[i, j]
                if prob >= threshold:
                    name_i = self.node_names[i]
                    name_j = self.node_names[j]
                    distance = 50.0 * (1 - prob) + 10.0  # Convert prob to distance
                    high_conf_edges.append((name_i, name_j, distance, prob))
        
        xml_content = '<?xml version="1.0"?>\n<restraints>\n'
        for name1, name2, distance, weight in high_conf_edges:
            xml_content += f'''  <distance_restraint>
    <particle1>{name1}</particle1>
    <particle2>{name2}</particle2>
    <distance>{distance:.2f}</distance>
    <weight>{weight:.3f}</weight>
    <sigma>5.0</sigma>
  </distance_restraint>
'''
        xml_content += '</restraints>\n'
        
        with open(output_file, 'w') as f:
            f.write(xml_content)
        
        logger.info(f"Saved {len(high_conf_edges)} restraints to {output_file}")


class Benchmarker:
    """Benchmark predicted interactions against PDB and CORUM."""
    
    def __init__(self, edge_dict: Dict[Tuple[str, str], float]):
        """
        Initialize benchmarker.
        
        Args:
            edge_dict: Dictionary mapping (protein1, protein2) -> probability
        """
        self.edge_dict = edge_dict
        self.predicted_pairs = set(
            tuple(sorted([p1, p2])) for p1, p2 in edge_dict.keys()
        )
    
    def load_pdb_interactions(self, pdb_dir: Path) -> set:
        """Load known interactions from PDB structures."""
        pdb_interactions = set()
        
        if not pdb_dir.exists():
            logger.warning(f"PDB directory not found: {pdb_dir}")
            return pdb_interactions
        
        # This would parse PDB files to extract protein-protein contacts
        # For now, return empty set
        logger.info("PDB interaction loading not yet implemented")
        return pdb_interactions
    
    def load_corum_interplexes(self, corum_file: Path) -> set:
        """Load known complexes from CORUM database."""
        corum_complexes = set()
        
        if not corum_file.exists():
            logger.warning(f"CORUM file not found: {corum_file}")
            return corum_complexes
        
        try:
            # CORUM format: tab-separated with complex subunits
            df = pd.read_csv(corum_file, sep='\t')
            if 'subunits' in df.columns:
                for _, row in df.iterrows():
                    subunits = str(row['subunits']).split(';')
                    # Create all pairs within complex
                    for i, sub1 in enumerate(subunits):
                        for sub2 in subunits[i+1:]:
                            corum_complexes.add(tuple(sorted([sub1.strip(), sub2.strip()])))
        except Exception as e:
            logger.error(f"Error loading CORUM: {e}")
        
        return corum_complexes
    
    def calculate_metrics(self, known_interactions: set, threshold: float = 0.5) -> Dict:
        """
        Calculate precision, recall, F1 for predictions.
        
        Args:
            known_interactions: Set of known (protein1, protein2) tuples
            threshold: Probability threshold for predictions
            
        Returns:
            Dictionary with metrics
        """
        # Get predicted pairs above threshold
        predicted = set(
            tuple(sorted([p1, p2])) 
            for (p1, p2), prob in self.edge_dict.items()
            if prob >= threshold
        )
        
        # Calculate metrics
        true_positives = len(predicted & known_interactions)
        false_positives = len(predicted - known_interactions)
        false_negatives = len(known_interactions - predicted)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'threshold': threshold,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'n_predicted': len(predicted),
            'n_known': len(known_interactions)
        }
    
    def benchmark(self, pdb_dir: Optional[Path] = None, 
                  corum_file: Optional[Path] = None,
                  thresholds: List[float] = [0.5, 0.6, 0.7, 0.8, 0.9]) -> pd.DataFrame:
        """
        Run full benchmark against PDB and CORUM.
        
        Returns:
            DataFrame with metrics at different thresholds
        """
        known_interactions = set()
        
        if pdb_dir:
            pdb_interactions = self.load_pdb_interactions(pdb_dir)
            known_interactions.update(pdb_interactions)
            logger.info(f"Loaded {len(pdb_interactions)} PDB interactions")
        
        if corum_file:
            corum_interactions = self.load_corum_interplexes(corum_file)
            known_interactions.update(corum_interactions)
            logger.info(f"Loaded {len(corum_interactions)} CORUM interactions")
        
        if len(known_interactions) == 0:
            logger.warning("No known interactions loaded. Cannot benchmark.")
            return pd.DataFrame()
        
        # Calculate metrics at different thresholds
        results = []
        for threshold in thresholds:
            metrics = self.calculate_metrics(known_interactions, threshold)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Convert MCMC results to IMP restraints and benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results', type=Path, required=True,
                       help='Path to MCMC results pickle file')
    parser.add_argument('--model-data', type=Path,
                       help='Path to model_data pickle file (auto-detected if not provided)')
    parser.add_argument('--output', type=Path, default='imp_restraints',
                       help='Output directory for restraints and scripts')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Probability threshold for high-confidence edges')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmarking against PDB/CORUM')
    parser.add_argument('--pdb-dir', type=Path,
                       help='Directory containing PDB structures')
    parser.add_argument('--corum-file', type=Path,
                       help='Path to CORUM database file')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Extract edge probabilities
    logger.info("Extracting edge probabilities from MCMC samples...")
    extractor = EdgeProbabilityExtractor(args.results, args.model_data)
    edge_probs, edge_dict = extractor.get_edge_probabilities()
    
    # Get node names
    node_names = list(extractor.node_idx2name.values()) if extractor.node_idx2name else \
                 [f"protein_{i}" for i in range(extractor.N)]
    
    # Save edge list
    edge_list_file = args.output / 'predicted_interactions.tsv'
    extractor.save_edge_list(edge_list_file, threshold=args.threshold)
    
    # Generate IMP restraints
    logger.info("Generating IMP restraints...")
    restraint_gen = IMPRestraintGenerator(edge_probs, node_names, edge_dict)
    
    # Generate Python script
    script_file = args.output / 'imp_modeling_script.py'
    restraint_gen.generate_restraints_script(script_file, threshold=args.threshold)
    
    # Generate XML restraints
    xml_file = args.output / 'restraints.xml'
    restraint_gen.save_restraints_xml(xml_file, threshold=args.threshold)
    
    # Benchmark if requested
    if args.benchmark:
        logger.info("Running benchmarks...")
        benchmarker = Benchmarker(edge_dict)
        metrics_df = benchmarker.benchmark(
            pdb_dir=args.pdb_dir,
            corum_file=args.corum_file
        )
        
        if not metrics_df.empty:
            metrics_file = args.output / 'benchmark_metrics.tsv'
            metrics_df.to_csv(metrics_file, sep='\t', index=False)
            logger.info(f"Benchmark metrics saved to {metrics_file}")
            print("\nBenchmark Results:")
            print(metrics_df.to_string(index=False))
    
    # Save summary
    summary = {
        'results_file': str(args.results),
        'n_proteins': extractor.N,
        'n_edges_above_threshold': len([p for p in edge_dict.values() if p >= args.threshold]),
        'high_confidence_edges': len(extractor.get_high_confidence_edges(args.threshold)),
        'output_files': {
            'edge_list': str(edge_list_file),
            'imp_script': str(script_file),
            'restraints_xml': str(xml_file)
        }
    }
    
    summary_file = args.output / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nIntegration complete! Files saved to {args.output}")
    logger.info(f"  - Edge list: {edge_list_file}")
    logger.info(f"  - IMP script: {script_file}")
    logger.info(f"  - Restraints XML: {xml_file}")
    logger.info(f"  - Summary: {summary_file}")


if __name__ == '__main__':
    main()

