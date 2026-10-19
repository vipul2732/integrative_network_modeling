# Integrative Network Modeling (INM)

INM infers a network of direct protein–protein contacts from affinity purification–mass spectrometry (AP-MS) data. Each network is an undirected graph on the proteins in the experiment. A continuous score ranks candidate networks by agreement with AP-MS spectral profiles, SAINT confidence scores, co-purification (composite) connectivity, and statistical priors on edge density and node degree. Alternative networks are sampled with the No-U-Turn Sampler (NUTS) in [NumPyro](https://num.pyro.ai/).

This repository is the Python implementation used for:

> Kumar V, Palar A, Krogan NJ, Echeverria I, Sali A. Modeling networks of direct protein–protein contacts based on affinity purification–mass spectrometry data.

The HIV-1 CRL5 AP-MS tables, analysis scripts, and deposited figures and tables are in the companion repository [salilab/inm-hiv-crl5](https://github.com/salilab/inm-hiv-crl5).


## Requirements

- Python 3.10 or 3.11 (3.10 is the tested version)
- Conda (Miniconda or Anaconda)
- A CUDA GPU is optional. Sampling runs on CPU with `--cpu`.

## Installation

From this directory:

```bash
conda env create -f environment.yml
conda activate inm-py310
```

CPU-only JAX is installed by default. For a CUDA 12 GPU:

```bash
pip install "jax[cuda12]" jax-cuda12-plugin
```

Check the install:

```bash
python -c "import numpyro, jax, pandas, openpyxl; print('JAX', jax.__version__, jax.devices())"
```

`setup_environment.sh` is an equivalent interactive installer.

Optional IMP export (`script/imp_integration.py`) needs a separate IMP installation (`conda install -c salilab imp`). Sampling does not require IMP.

## Input table

Provide an Excel workbook (`.xlsx`). Each sheet is one AP-MS table with at least these columns:

| Column | Meaning |
| --- | --- |
| `Bait` | Bait protein |
| `PreyGene` | Prey protein (the column name is configurable) |
| `SaintScore` | SAINT score |
| `Spec` | Spectral counts for the replicates, separated by `\|` (for example `10\|15\|12`) |
| `ctrlCounts` | Control spectral counts, same separator |

Rows with a missing or zero SAINT score are dropped. The manuscript runs read the first sheet only (`sheet_nums: 1`).

## Run a dataset

`examples/example_config.json` is a complete configuration. Edit `data.input_file` and `output.output_dir`, then from `script/`:

```bash
cd script
python run.py --config ../examples/example_config.json --no-figures
```

Paths inside a config are resolved relative to `script/`.

Command-line flags override the config:

```bash
python run.py \
  --data /path/to/table.xlsx \
  --output ../results/my_run \
  --model model23_n \
  --rseed 13 \
  --warmup 1000 \
  --samples 10000 \
  --no-figures \
  --cpu
```

`--cpu` forces JAX onto the CPU. Omit it when a GPU is available.

A short smoke test (100 warmup steps and 500 samples) confirms that preprocessing and sampling start:

```bash
python run_multiple_seeds.py \
  --config ../examples/example_config.json \
  --seeds 13 \
  --test \
  --cpu
```

The smoke test is not a manuscript run. Manuscript sampling uses 1,000 warmup steps and 10,000 sampling steps.

### Several chains

The manuscript uses independent chains with seeds 13, 42, and 71. Those three seeds are the default when `--n-runs 3` is set.

```bash
python run_multiple_seeds.py \
  --config ../examples/example_config.json \
  --seeds 13 42 71 \
  --output-dir ../results/my_dataset_multi_seeds \
  --cpu
```

Merge the chains and write convergence plots (score trace, R-hat, trace, autocorrelation) to `<output-dir>/merged/`:

```bash
python merge_and_analyze_convergence.py ../results/my_dataset_multi_seeds
```

Write the posterior mean probability of every protein pair:

```bash
python extract_all_pairs.py \
  --results ../results/my_dataset_multi_seeds/merged/merged_chains.pkl \
  --output ../results/my_dataset_multi_seeds/all_pairs.tsv
```

`all_pairs.tsv` has columns `Protein1`, `Protein2`, and `Probability`.

## Manuscript model settings

These are the settings in `examples/example_config.json` and in the HIV-1 CRL5 configs:

| Setting | Value |
| --- | --- |
| Model | `model23_n` |
| Warmup / samples | 1,000 / 10,000 per chain |
| Seeds | 13, 42, 71 |
| SAINT threshold ladder | 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 |
| Indirect SAINT | geometric mean of co-purified bait–prey scores, kept when both scores are at least 0.3 |
| Proteins modeled | top 80 by combined SAINT and multi-bait score (`max_proteins`, default 80). In the CRL5 data, 77 proteins passed the SAINT filter, giving 2,926 pairs |
| Consensus edge | posterior frequency at least 0.6 |

The score has five terms: AP-MS profile cosine similarity, a SAINT-derived interaction term, a composite-connectivity restraint, an edge-density prior, and a node-degree prior. Leave-one-term-out models are selected by the model name:

| Model name | Term removed |
| --- | --- |
| `model23_n_no_corr` | AP-MS profile similarity |
| `model23_n_no_saint` | SAINT term |
| `model23_n_no_priors` | degree and edge-density priors |

`script/run_ablated_models.py` launches those three models from a base config.

## Outputs of one chain

Inside the output directory:

- `config.json` — the resolved configuration
- `preprocessed/spec_table.tsv` — spectral-count matrix
- `preprocessed/composite_table.tsv` — SAINT and composite inputs
- `modeler_vars.json` — hyperparameters passed to the model
- `{model_id}_{model_name}_{seed}.pkl` — posterior samples
- `run.log`

The first JAX compilation for a new number of proteins is slow. Later chains with the same size reuse the compiled model.

## Model assessment helpers

These scripts are part of the method and are also used by the manuscript analyses:

- `merge_and_analyze_convergence.py` — pool chains and plot sampling diagnostics
- `sampling_precision_mixing.py` and `compute_post_sampling_imp_metrics.py` — sampling precision and model precision on the binary networks
- `validate_model_cost_perturbations.py` — node permutation, edge flipping, Gaussian noise, and edge removal
- `compare_community_detection.py` — Louvain communities
- `imp_integration.py` — optional export of high-probability edges as IMP restraints

## Citation

If you use this code, please cite the INM manuscript and NumPyro.

## License

MIT. See [LICENSE](LICENSE).
