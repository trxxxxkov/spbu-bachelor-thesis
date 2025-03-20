## Codebase structure

```bash
src/  # Main directory for the project's source code
├── data/          # Directory for datasets
│   └── README.md  # Local use warning
├── models/        # Directory for weights of the trained proposed models
│   └── README.md  # Local use warning
├── nn_modules/             # Custom PyTorch layers for the proposed models
│   ├── ensemble_kan.py     # 
│   ├── kan.py              # Custom PyTorch implementation of a KAN layer
│   ├── regularized_kan.py  #
│   ├── rehearsal_kan.py    #
│   └── sparse_kan.py       #
├── notebooks/                     # Directory for experiments and visualization
│   ├── 01_data_preparation.ipynb  # Datasets downloading and extraction of embeddings
│   └── 02_baseline_models.ipynb   # Baseline models (KAN and MLP) offline training
├── utils/                   # Directory for miscellaneous convenience functions
│   ├── datasets.py          # Custom PyTorch datasets and data-related utilities
│   ├── global_constants.py  # Constants and certain params mentioned in the thesis  
│   └── visualization.py     # Custom functions for plots and visualization
└── README.md  # [this] Documentation for the source code directory structure
```