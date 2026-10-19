#!/usr/bin/env python3
"""
Add UniProt IDs to protein pair TSV files (robust batch mapping).

What this script does
- Reads an input TSV with columns: Protein1, Protein2, Probability (others allowed)
- Maps unique protein/gene names -> UniProt accessions using mygene (batch querymany)
- Adds Protein1_UniProt and Protein2_UniProt columns
- Saves:
  (1) updated TSV
  (2) mapping JSON (protein name -> UniProt or null)
  (3) optional unmapped list TSV for quick debugging

Key improvements vs your version
- Uses querymany() (batch) with multiple scopes (symbol/alias/name)
- Handles UniProt field shapes robustly (Swiss-Prot / TrEMBL as str or list)
- Normalizes input names (strip); avoids empty-string “false mapped” counts
- Saves the authoritative mapping dict (no fragile reconstruction from df)

Usage
  python add_uniprot_ids_v2.py \
    --input results/cullin_wt_config_multi_seeds/all_pairs.tsv \
    --output results/cullin_wt_config_multi_seeds/all_pairs_with_uniprot.tsv \
    --mapping results/cullin_wt_config_multi_seeds/protein_to_uniprot.json \
    --unmapped results/cullin_wt_config_multi_seeds/unmapped_proteins.tsv

  # For all cullin datasets:
  for dataset in wt mock vif; do
    python add_uniprot_ids_v2.py \
      --input results/cullin_${dataset}_config_multi_seeds/all_pairs.tsv \
      --output results/cullin_${dataset}_config_multi_seeds/all_pairs_with_uniprot.tsv \
      --mapping results/cullin_${dataset}_config_multi_seeds/protein_to_uniprot.json \
      --unmapped results/cullin_${dataset}_config_multi_seeds/unmapped_proteins.tsv
  done
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple

import pandas as pd

try:
    import mygene
except ImportError as e:
    raise SystemExit("ERROR: mygene not installed. Install with: pip install mygene") from e

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------
def normalize_name(x: Any) -> str:
    """Normalize protein/gene name keys used in mapping."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def is_missing_uniprot(x: Any) -> bool:
    """Treat NaN/None/empty-string as missing UniProt."""
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def pick_uniprot_from_hit(hit: Dict[str, Any]) -> Optional[str]:
    """
    Extract a single UniProt accession from a mygene hit.
    Prefer Swiss-Prot (reviewed) then TrEMBL.
    Handles both string and list forms.
    """
    u = hit.get("uniprot")
    if not isinstance(u, dict):
        return None

    sp = u.get("Swiss-Prot")
    tr = u.get("TrEMBL")

    if isinstance(sp, list):
        return sp[0] if sp else None
    if isinstance(sp, str):
        return sp.strip() or None

    if isinstance(tr, list):
        return tr[0] if tr else None
    if isinstance(tr, str):
        return tr.strip() or None

    return None


def is_special_skip(name: str) -> bool:
    """
    Skip names that are not expected to map to human UniProt.
    Customize as needed.
    """
    n = name.strip().lower()
    return n in {"vifprotein", "vif"}  # keep your special-case


# ----------------------------
# Mapping core
# ----------------------------
def load_uniprot_from_input_files(dataset_name: str) -> Dict[str, Optional[str]]:
    """
    Load UniProt IDs from input Excel files (cullin_wt.xlsx, etc.).
    Maps PreyGene -> UniProt ID (from Prey column).
    """
    mapping = {}
    
    # Try to find the input Excel file
    possible_paths = [
        Path(f"input/cullin/cullin_{dataset_name}/cullin_{dataset_name}.xlsx"),
        Path(f"input/cullin/{dataset_name}/cullin_{dataset_name}.xlsx"),
    ]
    
    xlsx_file = None
    for path in possible_paths:
        if path.exists():
            xlsx_file = path
            break
    
    if not xlsx_file:
        logger.debug(f"No input Excel file found for {dataset_name}")
        return mapping
    
    try:
        import openpyxl
        df = pd.read_excel(xlsx_file, engine='openpyxl')
        
        if 'PreyGene' not in df.columns or 'Prey' not in df.columns:
            logger.debug(f"Excel file missing PreyGene or Prey columns: {xlsx_file}")
            return mapping
        
        # Create mapping: PreyGene -> UniProt ID (from Prey column)
        # Prey column contains UniProt IDs (like 'Q9UBF6', 'Q93034')
        for _, row in df.iterrows():
            prey_gene = str(row.get('PreyGene', '')).strip()
            prey_uniprot = str(row.get('Prey', '')).strip()
            
            # Skip if either is missing or NaN
            if pd.isna(row.get('PreyGene')) or pd.isna(row.get('Prey')):
                continue
            
            # Check if Prey looks like a UniProt ID (starts with letter, followed by alphanumeric)
            # UniProt IDs are typically like: Q9UBF6, P12345, etc.
            if prey_uniprot and len(prey_uniprot) >= 6 and prey_uniprot[0].isalpha():
                # Normalize gene name - try both with and without _HUMAN suffix
                prey_gene_norm = normalize_name(prey_gene)
            if prey_gene_norm:
                # Map with original name
                if prey_gene_norm not in mapping or mapping[prey_gene_norm] is None:
                    mapping[prey_gene_norm] = prey_uniprot.upper()
                
                # Also map without _HUMAN suffix if present
                if prey_gene_norm.endswith('_HUMAN'):
                    prey_gene_no_suffix = prey_gene_norm[:-6]  # Remove '_HUMAN'
                    if prey_gene_no_suffix not in mapping or mapping[prey_gene_no_suffix] is None:
                        mapping[prey_gene_no_suffix] = prey_uniprot.upper()
                
                # Also map with _HUMAN suffix if original doesn't have it
                if not prey_gene_norm.endswith('_HUMAN'):
                    prey_gene_with_suffix = prey_gene_norm + '_HUMAN'
                    if prey_gene_with_suffix not in mapping or mapping[prey_gene_with_suffix] is None:
                        mapping[prey_gene_with_suffix] = prey_uniprot.upper()
        
        logger.info(f"Loaded {len([v for v in mapping.values() if v])} UniProt mappings from {xlsx_file}")
        
    except Exception as e:
        logger.warning(f"Error loading Excel file {xlsx_file}: {e}")
    
    return mapping


def map_proteins_to_uniprot_batch(
    proteins: List[str],
    species: str = "human",
    batch_size: int = 1000,
    sleep_s: float = 0.2,
    retry_count: int = 3,
) -> Dict[str, Optional[str]]:
    """
    Batch map protein/gene names to UniProt accessions using mygene.querymany.

    Returns mapping dict: original_name -> UniProt or None
    """
    mg = mygene.MyGeneInfo()

    # Ensure stable, clean list
    proteins = [normalize_name(p) for p in proteins]
    proteins = [p for p in proteins if p]
    proteins = sorted(set(proteins))

    mapping: Dict[str, Optional[str]] = {}

    # Pre-fill special skip names
    for p in proteins:
        if is_special_skip(p):
            mapping[p] = None

    query_list = [p for p in proteins if p not in mapping]  # exclude pre-filled
    logger.info(f"Mapping {len(query_list)} proteins via mygene (species={species})")

    # Chunked querymany for robustness on very large sets
    for start in range(0, len(query_list), batch_size):
        chunk = query_list[start : start + batch_size]
        logger.info(f"Querying chunk {start//batch_size + 1} / {(len(query_list)-1)//batch_size + 1} "
                    f"(n={len(chunk)})")

        last_err = None
        for attempt in range(1, retry_count + 1):
            try:
                results = mg.querymany(
                    chunk,
                    scopes=["symbol", "alias", "name"],
                    fields=["uniprot.Swiss-Prot", "uniprot.TrEMBL"],
                    species=species,
                    as_dataframe=False,
                    returnall=False,
                    verbose=False,
                )
                last_err = None
                break
            except Exception as e:
                last_err = e
                wait = attempt * 1.0
                logger.warning(f"mygene querymany error (attempt {attempt}/{retry_count}): {e} | waiting {wait}s")
                time.sleep(wait)

        if last_err is not None:
            logger.error(f"Failed chunk after {retry_count} retries; marking as unmapped. Error: {last_err}")
            for p in chunk:
                mapping[p] = None
            continue

        # Parse results. Each result has at least 'query' and may have 'notfound'
        for hit in results:
            q = normalize_name(hit.get("query", ""))
            if not q:
                continue
            if hit.get("notfound", False):
                mapping[q] = None
                continue
            mapping[q] = pick_uniprot_from_hit(hit)

        time.sleep(sleep_s)

    # Ensure every protein has a key
    for p in proteins:
        mapping.setdefault(p, None)

    mapped = sum(1 for v in mapping.values() if not is_missing_uniprot(v))
    logger.info(f"Mapping complete: {mapped}/{len(mapping)} proteins mapped")
    return mapping


def load_mapping_json(path: Path) -> Dict[str, Optional[str]]:
    with open(path, "r") as f:
        raw = json.load(f)
    # normalize keys + values
    out: Dict[str, Optional[str]] = {}
    for k, v in raw.items():
        kk = normalize_name(k)
        if not kk:
            continue
        out[kk] = None if is_missing_uniprot(v) else str(v).strip()
    return out


def save_mapping_json(path: Path, mapping: Dict[str, Optional[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # ensure JSON-serializable (None or string)
    clean = {k: (None if is_missing_uniprot(v) else str(v).strip()) for k, v in mapping.items()}
    with open(path, "w") as f:
        json.dump(clean, f, indent=2, sort_keys=True)


# ----------------------------
# Main TSV augmentation
# ----------------------------
def add_uniprot_ids_to_tsv(
    input_file: Path,
    output_file: Path,
    mapping_file: Optional[Path] = None,
    unmapped_file: Optional[Path] = None,
    species: str = "human",
    remap_missing: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Optional[str]]]:
    logger.info(f"Loading input TSV: {input_file}")
    df = pd.read_csv(input_file, sep="\t")

    for col in ["Protein1", "Protein2"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {input_file}")

    # Normalize names in dataframe (strip whitespace)
    df["Protein1"] = df["Protein1"].apply(normalize_name)
    df["Protein2"] = df["Protein2"].apply(normalize_name)

    unique_proteins = sorted(set(df["Protein1"].dropna().tolist()) | set(df["Protein2"].dropna().tolist()))
    unique_proteins = [p for p in unique_proteins if p]
    logger.info(f"Found {len(unique_proteins)} unique protein names")

    mapping: Dict[str, Optional[str]] = {}

    # First, try to load from input Excel files
    # Extract dataset name from input file path
    dataset_name = None
    input_str = str(input_file)
    if 'cullin_wt' in input_str:
        dataset_name = 'wt'
    elif 'cullin_mock' in input_str:
        dataset_name = 'mock'
    elif 'cullin_vif' in input_str:
        dataset_name = 'vif'
    
    if dataset_name:
        excel_mapping = load_uniprot_from_input_files(dataset_name)
        mapping.update(excel_mapping)
        excel_count = len([v for v in excel_mapping.values() if v])
        logger.info(f"Loaded {excel_count} UniProt mappings from input Excel file")

    # Load existing mapping if provided (will override Excel mappings if conflicts)
    if mapping_file and mapping_file.exists():
        logger.info(f"Loading existing mapping: {mapping_file}")
        existing_mapping = load_mapping_json(mapping_file)
        # Update mapping, but don't overwrite Excel mappings with None values
        for k, v in existing_mapping.items():
            if k not in mapping or mapping[k] is None:
                mapping[k] = v
        logger.info(f"Loaded {len(existing_mapping)} additional mapping entries")

    # Determine what to map
    if remap_missing:
        to_map = [p for p in unique_proteins if p not in mapping or is_missing_uniprot(mapping.get(p))]
    else:
        to_map = [p for p in unique_proteins if p not in mapping]

    if to_map:
        new_map = map_proteins_to_uniprot_batch(to_map, species=species)
        mapping.update(new_map)

    # Add UniProt columns
    df["Protein1_UniProt"] = df["Protein1"].map(mapping)
    df["Protein2_UniProt"] = df["Protein2"].map(mapping)

    # Save updated TSV (keep original columns; just insert UniProt cols after Protein1/2 if present)
    cols = list(df.columns)
    # Try to place UniProt cols right after Protein1/Protein2
    def insert_after(col_list, after_col, new_col):
        if new_col in col_list:
            col_list.remove(new_col)
        if after_col in col_list:
            idx = col_list.index(after_col) + 1
            col_list.insert(idx, new_col)
        else:
            col_list.append(new_col)

    insert_after(cols, "Protein1", "Protein1_UniProt")
    insert_after(cols, "Protein2", "Protein2_UniProt")
    df = df[cols]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Saved updated TSV: {output_file}")

    # Save mapping file
    if mapping_file:
        save_mapping_json(mapping_file, mapping)
        logger.info(f"Saved mapping JSON: {mapping_file}")

    # Save unmapped pairs if requested
    if unmapped_file:
        # Find pairs where at least one protein is missing UniProt ID
        p1_ok = ~df["Protein1_UniProt"].apply(is_missing_uniprot)
        p2_ok = ~df["Protein2_UniProt"].apply(is_missing_uniprot)
        unmapped_mask = ~(p1_ok & p2_ok)  # Pairs where at least one is missing
        
        unmapped_pairs = df[unmapped_mask].copy()
        unmapped_file.parent.mkdir(parents=True, exist_ok=True)
        unmapped_pairs.to_csv(unmapped_file, sep="\t", index=False)
        logger.info(f"Saved unmapped pairs: {unmapped_file} (n={len(unmapped_pairs)} pairs)")

    # Stats (robust to empty strings)
    p1_ok = ~df["Protein1_UniProt"].apply(is_missing_uniprot)
    p2_ok = ~df["Protein2_UniProt"].apply(is_missing_uniprot)

    total_pairs = len(df)
    both = int((p1_ok & p2_ok).sum())
    one = int(((p1_ok | p2_ok) & ~(p1_ok & p2_ok)).sum())
    none = int((~p1_ok & ~p2_ok).sum())

    logger.info(f"Pairs: total={total_pairs} | both_uniprot={both} | one_uniprot={one} | none_uniprot={none}")

    return df, mapping


def main():
    ap = argparse.ArgumentParser(description="Add UniProt IDs to protein pair TSV files (robust batch mapping)")
    ap.add_argument("--input", type=Path, required=True, help="Input TSV file with Protein1/Protein2")
    ap.add_argument("--output", type=Path, required=True, help="Output TSV with UniProt columns added")
    ap.add_argument("--mapping", type=Path, default=None, help="JSON mapping file (load if exists; save updated)")
    ap.add_argument("--unmapped", type=Path, default=None, help="Output TSV file for pairs without UniProt IDs")
    ap.add_argument("--species", type=str, default="human", help="Species for mygene mapping (default: human)")
    ap.add_argument(
        "--no-remap-missing",
        action="store_true",
        help="If set, do not remap names that exist in mapping JSON but are missing/None",
    )

    args = ap.parse_args()

    add_uniprot_ids_to_tsv(
        input_file=args.input,
        output_file=args.output,
        mapping_file=args.mapping,
        unmapped_file=args.unmapped,
        species=args.species,
        remap_missing=not args.no_remap_missing,
    )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

