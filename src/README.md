## Codebase structure
```bash
src/  # Main directory for the project's source code
├── data/          # Directory for data-related files
│   └── README.md  # Local use warning
├── models/                 # Directory containing model implementations
│   ├── ensemble_kan.py     # Implementation of the ensemble KAN model
│   ├── regularized_kan.py  # Implementation of the regularized KAN model
│   ├── rehearsal_kan.py    # Implementation of the rehearsal KAN model
│   └── sparse_kan.py       # Implementation of the sparse KAN model
├── notebooks/                     # Directory for experiments and visualization
│   └── 01_data_preparation.ipynb  # Datasets downloading and preprocessing
├── trained_models/  # Directory for storing trained model files
│   └── README.md    # Local use warning
├── utils/                # Directory for utility scripts
│   └── visualization.py  # Scripts for plots and data visualization
└── README.md  # Documentation for the source code directory structure
```