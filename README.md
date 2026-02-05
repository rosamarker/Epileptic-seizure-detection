### Overview

This repository contains Python scripts for preparing EEG and ECG datasets, training deep learning models, and evaluating seizure and pre-ictal detection performance.
The codebase focuses on practical experiment workflows including data integrity checks, split creation, generator based loading, metric reporting, and figure generation.

### Repository Contents

- Main training entrypoints are provided in `main_func.py`, `main_net.py`, and related configuration files
- Data loading and generator logic is implemented in `generator_ds.py` and `generator_ds_backup.py`
- Model definitions include `ChronoNet.py` and supporting modules
- Evaluation and reporting utilities include `run_evaluate.py`, `performance.py`, and `cm.py`
- Dataset preparation include `make_split.py`, `make_SZ2_all_subjects.py`, `fix_symlinks.py`, `check_edfs.py`, `missing_signals.py`, and `check_preictal_testset_fixed.py`
- Plotting helpers include `plot_loss_accuracy.py`, `plot_tools.py`, and `fa-h.py`
- General helpers and shared routines are provided in `utils.py` and `routines.py`
- GPU and environment helpers are provided in `gpu.py` and `environment.yml`  ￼

### Environment

- Python version is 3.10  ￼
- Install the depencies as the following:
  ```bash
  cd /path/to/repo
  conda env create -f environment.yml
  conda activate <env_name>
  ```

### Typical Workflow

- Validate raw EDF files and signal availability using `check_edfs.py` and `missing_signals.py`
- Create or verify subject level splits using `make_split.py` and related split utilities
- Train a model using `main_net.py` with the desired configuration
- Evaluate trained checkpoints using `run_evaluate.py` and `performance.py`
- Generate plots for learning curves and result summaries using `plot_loss_accuracy.py` and `plot_tools.py`

### Data Expectations

- Input data is expected to be available as EDF files readable by `pyedflib` ￼
- Splits and metadata are produced by the split generation scripts and consumed by the generator modules
- Paths, labels, and sampling conventions are controlled through configuration files in the repository

### Reproducibility Notes

- Keep splits fixed for fair comparisons across model variants
- Track configuration files alongside saved checkpoints for experiment traceability
- Use the same preprocessing and windowing settings across training and evaluation runs
