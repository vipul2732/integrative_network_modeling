#!/usr/bin/env python3
"""
Interactive script to select Excel sheets for processing.

This script helps users:
1. View available sheets in an Excel file
2. Select which sheets to process (1, 3, or any number)
3. Preview sheet contents
4. Generate a config file for use with run.py

Usage:
    python select_sheets.py --data input/cullin/1-s2.0-S1931312819302537-mmc2.xlsx
    python select_sheets.py --data input/cullin/1-s2.0-S1931312819302537-mmc2.xlsx --sheets 0 1 2
    python select_sheets.py --data input/cullin/1-s2.0-S1931312819302537-mmc2.xlsx --interactive
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required. Install with: pip install pandas openpyxl")
    sys.exit(1)


def list_sheets(xlsx_path: Path) -> List[str]:
    """List all sheet names in the Excel file."""
    try:
        excel_file = pd.ExcelFile(xlsx_path)
        return excel_file.sheet_names
    except Exception as e:
        print(f"ERROR: Could not read Excel file: {e}")
        sys.exit(1)


def preview_sheet(xlsx_path: Path, sheet_name: str, n_rows: int = 5) -> pd.DataFrame:
    """Preview first few rows of a sheet."""
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, nrows=n_rows)
        return df
    except Exception as e:
        print(f"WARNING: Could not preview sheet '{sheet_name}': {e}")
        return pd.DataFrame()


def validate_sheet(xlsx_path: Path, sheet_name: str, 
                   required_cols: List[str] = None) -> tuple:
    """Validate that a sheet has required columns."""
    if required_cols is None:
        required_cols = ['Bait', 'Prey', 'Spec', 'ctrlCounts']
    
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, nrows=1)
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            return False, f"Missing columns: {', '.join(missing_cols)}"
        
        # Check if sheet has data
        df_full = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        if len(df_full) == 0:
            return False, "Sheet is empty"
        
        return True, "OK"
    except Exception as e:
        return False, f"Error: {str(e)}"


def interactive_sheet_selection(xlsx_path: Path) -> List[int]:
    """Interactively select sheets from the Excel file."""
    sheet_names = list_sheets(xlsx_path)
    
    print("\n" + "=" * 60)
    print(f"Excel file: {xlsx_path.name}")
    print(f"Total sheets: {len(sheet_names)}")
    print("=" * 60)
    print("\nAvailable sheets:")
    print("-" * 60)
    
    for i, name in enumerate(sheet_names):
        # Quick validation
        is_valid, msg = validate_sheet(xlsx_path, name)
        status = "✓" if is_valid else "✗"
        
        # Preview columns
        try:
            df_preview = preview_sheet(xlsx_path, name, n_rows=1)
            cols = ", ".join(df_preview.columns[:5].tolist())
            if len(df_preview.columns) > 5:
                cols += "..."
        except:
            cols = "N/A"
        
        print(f"  [{i}] {name:30s} {status:2s} | Columns: {cols}")
        if not is_valid:
            print(f"      └─ {msg}")
    
    print("\n" + "-" * 60)
    
    while True:
        try:
            selection = input(
                "\nEnter sheet numbers to use (comma-separated, e.g., 0,1,2 or 'all'): "
            ).strip()
            
            if selection.lower() == 'all':
                return list(range(len(sheet_names)))
            
            # Parse selection
            indices = [int(x.strip()) for x in selection.split(',')]
            
            # Validate indices
            invalid = [i for i in indices if i < 0 or i >= len(sheet_names)]
            if invalid:
                print(f"ERROR: Invalid sheet numbers: {invalid}")
                print(f"Please enter numbers between 0 and {len(sheet_names)-1}")
                continue
            
            # Validate selected sheets
            invalid_sheets = []
            for i in indices:
                is_valid, msg = validate_sheet(xlsx_path, sheet_names[i])
                if not is_valid:
                    invalid_sheets.append((i, sheet_names[i], msg))
            
            if invalid_sheets:
                print("\nWARNING: Some selected sheets have issues:")
                for i, name, msg in invalid_sheets:
                    print(f"  Sheet [{i}] {name}: {msg}")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            return sorted(set(indices))  # Remove duplicates and sort
            
        except ValueError:
            print("ERROR: Please enter valid numbers separated by commas")
        except KeyboardInterrupt:
            print("\n\nCancelled by user")
            sys.exit(0)


def create_config_file(
    xlsx_path: Path,
    selected_sheets: List[int],
    output_config: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Path:
    """Create a config file with selected sheets."""
    sheet_names = list_sheets(xlsx_path)
    selected_sheet_names = [sheet_names[i] for i in selected_sheets]
    
    # Determine output directory
    if output_dir is None:
        # Use same directory as Excel file, or create a results directory
        if xlsx_path.parent.name == 'input':
            output_dir = xlsx_path.parent.parent / 'results' / xlsx_path.stem
        else:
            output_dir = xlsx_path.parent / 'results' / xlsx_path.stem
    else:
        output_dir = Path(output_dir)
    
    # Create config
    config = {
        'data': {
            'input_file': str(xlsx_path),
            'prey_column': 'PreyGene',
            'filter': 'all',
            'enforce_bait_remapping': True,
            'sheet_nums': len(selected_sheets),  # Number of sheets to process
            'selected_sheet_indices': selected_sheets,  # Which sheets (0-indexed)
            'selected_sheet_names': selected_sheet_names,  # Sheet names for reference
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
            'output_dir': str(output_dir),
            'save_warmup': True,
            'include_extra_fields': True,
        },
        'figures': {
            'generate': True,
        },
        'imp': {
            'generate': False,
            'threshold': 0.7,
            'benchmark': False,
            'pdb_dir': None,
            'corum_file': None,
        }
    }
    
    # Determine config file path
    if output_config is None:
        config_file = xlsx_path.parent / f"{xlsx_path.stem}_config.json"
    else:
        config_file = Path(output_config)
    
    # Save config
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_file


def main():
    parser = argparse.ArgumentParser(
        description='Select Excel sheets for processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive selection
  python select_sheets.py --data input/cullin/data.xlsx --interactive
  
  # Select specific sheets by index
  python select_sheets.py --data input/cullin/data.xlsx --sheets 0 1 2
  
  # Select first 3 sheets
  python select_sheets.py --data input/cullin/data.xlsx --sheets 0 1 2 --output-config my_config.json
  
  # Just list sheets without selecting
  python select_sheets.py --data input/cullin/data.xlsx --list-only
        """
    )
    
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to Excel file')
    parser.add_argument('--sheets', type=int, nargs='+',
                       help='Sheet indices to select (0-indexed, e.g., 0 1 2)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive sheet selection')
    parser.add_argument('--list-only', action='store_true',
                       help='Only list sheets, do not create config')
    parser.add_argument('--output-config', type=Path,
                       help='Output config file path (default: <excel_name>_config.json)')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for results (default: results/<excel_name>)')
    
    args = parser.parse_args()
    
    # Resolve Excel file path
    xlsx_path = Path(args.data)
    if not xlsx_path.is_absolute():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        xlsx_path = (script_dir / xlsx_path).resolve()
        # Also try from input directory
        if not xlsx_path.exists():
            workflow_dir = script_dir.parent
            alt_path = workflow_dir / 'input' / args.data.name
            if alt_path.exists():
                xlsx_path = alt_path
    
    if not xlsx_path.exists():
        print(f"ERROR: Excel file not found: {xlsx_path}")
        sys.exit(1)
    
    # List sheets
    sheet_names = list_sheets(xlsx_path)
    
    print("\n" + "=" * 60)
    print(f"Excel file: {xlsx_path}")
    print(f"Total sheets: {len(sheet_names)}")
    print("=" * 60)
    
    if args.list_only:
        print("\nAvailable sheets:")
        for i, name in enumerate(sheet_names):
            is_valid, msg = validate_sheet(xlsx_path, name)
            status = "✓" if is_valid else "✗"
            print(f"  [{i}] {name:40s} {status:2s}")
            if not is_valid:
                print(f"      └─ {msg}")
        return
    
    # Select sheets
    if args.interactive:
        selected_indices = interactive_sheet_selection(xlsx_path)
    elif args.sheets:
        # Validate indices
        invalid = [i for i in args.sheets if i < 0 or i >= len(sheet_names)]
        if invalid:
            print(f"ERROR: Invalid sheet indices: {invalid}")
            print(f"Valid indices are 0 to {len(sheet_names)-1}")
            sys.exit(1)
        selected_indices = sorted(set(args.sheets))
    else:
        # Default: use first 3 sheets (or all if less than 3)
        selected_indices = list(range(min(3, len(sheet_names))))
        print(f"\nNo selection specified. Using first {len(selected_indices)} sheet(s):")
        for i in selected_indices:
            print(f"  [{i}] {sheet_names[i]}")
    
    # Show selection summary
    print("\n" + "=" * 60)
    print("Selected sheets:")
    print("-" * 60)
    for i in selected_indices:
        is_valid, msg = validate_sheet(xlsx_path, sheet_names[i])
        status = "✓" if is_valid else "✗"
        print(f"  [{i}] {sheet_names[i]:40s} {status:2s}")
        if not is_valid:
            print(f"      └─ WARNING: {msg}")
    print("=" * 60)
    
    # Create config file
    config_file = create_config_file(
        xlsx_path,
        selected_indices,
        output_config=args.output_config,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Config file created: {config_file}")
    print(f"\nNext steps:")
    print(f"  # Single run:")
    print(f"  python run.py --config {config_file}")
    print(f"  ")
    print(f"  # Multiple runs with different seeds:")
    print(f"  python run_multiple_seeds.py --config {config_file} --n-runs 3")
    print(f"  python run_multiple_seeds.py --config {config_file} --seeds 13 42 99")
    print()


if __name__ == '__main__':
    main()

