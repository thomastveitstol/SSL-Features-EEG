# SSL-Features-EEG

![Tests](https://github.com/thomastveitstol/SSL-Features-EEG/actions/workflows/tests.yml/badge.svg)

A project for self-supervised learning on EEG data. The core idea is to learn patterns of healthy transitioning between the eyes-closed and eyes-open conditions at rest, which are also sensitive to pathological abnormalities. The code (and corresponding paper) was the third and last project of my Ph.D. However, it was intentionally designed to be straightforward for further work, such as adding new architectures, datasets, gradient-based methods for multi-objective optimisation, etc. 

The paper is available as a preprint.

doi: https://doi.org/10.1101/2025.09.21.677598

## Repository structure
The code is organised as follows:
- **`scripts/`** - All executable scripts. Key directories include:
  - `model_training/` - Includes scripts that were submitted for running experiments on Colossus in TSD, as well as configuration files.
  - `compute_pseudo_targets/` - Includes script that was used to compute the pseudo-targets. Note: the log-transform is applied at runtime, not during the computation of band-power values
  - `prepare_input_data/` - Includes script that was used for pre-processing the EEG data. The data is stored in multiple versions to make data loading more efficient (no need for, e.g., interpolation at runtime). 
- **`src/elecssl/`** - Contains core code, including classes and functions for data processing and model implementation. Key directories include:
  - `data/` - Includes classes and functions related to the data, such as pre-processing, data generators, and data splitting. A particularly used class is the `Subject` class in `subject_split/`, which defines a subject by both the dataset name and the subject ID. This is convenient as different EEG datasets sometimes use the same naming conventions, and hence this class avoids subject-ID conflicts.
  - `models/` - Includes classes and functions related to the models and DL methodology, including DL architectures, region based pooling, hyperparameter sampling, performance tracking, loss-functions, algorithms for multi-task learning/multi-objective optimisation, and the main experiments. In particular, the classes for running the hyperparameter optimisation experiments are found in `experiments/hpo_experiments`
- **`tests/`** - Contains tests for code in `src/elecssl/`.

## Notes
- Although this code was primarily developed for the eyes-closed/eyes-open transitions, the methods and code can easily be adapted for other stimulations as well. Although this implementation has currently not been made, it should not be too much work (and feel free to hit me up for guidance if you're interested)!
- This code used a former repo of mine as a starting point, which is located at https://github.com/thomastveitstol/CrossDatasetLearningEEG. For that reason, some unused code (such as unused datasets and domain adaptation methods) may still be present.

## Citation
```bibtex
@article {TveitstoelSSL2025,
	author = {Tveitstøl, Thomas and Tveter, Mats and Hatlestad-Hall, Christoffer and Hammer, Hugo  L. and Hebold Haraldsen, Ira R J},
	title = {Learning Residual-based Biomarkers of Cognitive Health via Self-Supervised Learning on EEG State Transitions},
	elocation-id = {2025.09.21.677598},
	year = {2025},
	doi = {10.1101/2025.09.21.677598},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/09/21/2025.09.21.677598},
	eprint = {https://www.biorxiv.org/content/early/2025/09/21/2025.09.21.677598.full.pdf},
	journal = {bioRxiv}
}
