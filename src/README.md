## Codebase structure

```bash
src/  # Main directory for the project's source code.
├── nn_modules/             # Custom PyTorch layers for the proposed models.
│   ├── ensemble_kan.py     # 
│   ├── kan.py              # Custom PyTorch implementation of a KAN layer.
│   ├── mlp.py              # Baseline model for CL.
│   ├── regularized_kan.py  #
│   ├── rehearsal_kan.py    #
│   └── sparse_kan.py       #
├── notebooks/                              # Directory for experiments and visualization.
│   ├── 01_data_preparation.ipynb           # Datasets downloading and extraction of embeddings.
│   ├── 02_proposed_models.ipynb            # Offline training and benchmarks for proposed models.
│   └── 03_continual_learning.ipynb         # CIL and data permutation experiments
├── utils/                   # Directory for miscellaneous convenience functions.
│   ├── datasets.py          # Custom PyTorch datasets and data-related utilities.
│   ├── global_constants.py  # Constants and certain params mentioned in the thesis.
│   ├── metrics.py           # Custom metrics classes and related functions.
│   ├── train_test.py        # Instruments for managing the train-test loop.
│   └── visualization.py     # Utility functions for plots and visualization.
└── README.md  # [this] Documentation for the source code directory structure.
```